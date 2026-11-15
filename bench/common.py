import json
import os
import platform
import time
from pathlib import Path

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DiffusionGemmaForBlockDiffusion,
    Gemma4ForConditionalGeneration,
)

from dgemma_tpu.dist import initialize_process_group, shard_experts
from dgemma_tpu.moe import register_experts_implementations
from dgemma_tpu.runtime import move_preserving_ties

DIFFUSION_MODEL_ID = "google/diffusiongemma-26B-A4B-it"
GEMMA4_MODEL_ID = "google/gemma-4-26B-A4B-it"
TINY_MODEL_ID = "trl-internal-testing/tiny-DiffusionGemmaForBlockDiffusion"
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def load_model(
    family: str,
    spec: str,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str | None = None,
    experts_implementation: str | None = None,
    expert_parallel: bool = False,
):
    """
    Loads a DiffusionGemma or Gemma 4 model for benchmarking.

    Args:
        family (`str`):
            `"diffusion"` for DiffusionGemma or `"gemma4"` for the autoregressive model.
        spec (`str`):
            `"proxy:<num_layers>"` for a randomly initialized model with the released width and a reduced depth,
            `"tiny"` for the DiffusionGemma test checkpoint, or a Hub id or local path of pretrained weights.
        device (`torch.device`):
            Device to place the model on.
        dtype (`torch.dtype`):
            Dtype of the weights.
        attn_implementation (`str`, *optional*):
            Attention implementation override.
        experts_implementation (`str`, *optional*):
            Mixture-of-experts implementation override.
        expert_parallel (`bool`, *optional*, defaults to `False`):
            Whether to slice the experts across the ranks of a `torchrun` launch and all-reduce their outputs.

    Returns:
        `PreTrainedModel`: The model in evaluation mode.
    """
    model_class = DiffusionGemmaForBlockDiffusion if family == "diffusion" else Gemma4ForConditionalGeneration
    released_model_id = DIFFUSION_MODEL_ID if family == "diffusion" else GEMMA4_MODEL_ID
    register_experts_implementations()
    overrides = {}
    if attn_implementation is not None:
        overrides["attn_implementation"] = attn_implementation
    if experts_implementation is not None:
        overrides["experts_implementation"] = experts_implementation

    if spec.startswith("proxy:"):
        config = proxy_config(released_model_id, int(spec.split(":", 1)[1]))
        for name, value in overrides.items():
            setattr(config, f"_{name}", value)
        model = model_class(config).to(dtype)
        return place_model(model, device, expert_parallel)

    model_id = TINY_MODEL_ID if spec == "tiny" else spec
    model = model_class.from_pretrained(model_id, dtype=dtype, **overrides)
    return place_model(model, device, expert_parallel)


def place_model(model, device: torch.device, expert_parallel: bool):
    """
    Shards the experts when requested and moves the model to its device without duplicating tied weights.

    Args:
        model (`PreTrainedModel`):
            Model with its weights on the host.
        device (`torch.device`):
            Target device.
        expert_parallel (`bool`):
            Whether to slice the experts across the ranks of the launch.

    Returns:
        `PreTrainedModel`: The model in evaluation mode on `device`.
    """
    if expert_parallel:
        rank, world_size = initialize_process_group(device.type)
        shard_experts(model, rank, world_size)
    return move_preserving_ties(model, device).eval()


def proxy_config(released_model_id: str, num_layers: int):
    """
    Builds a released configuration truncated to `num_layers` layers.

    Args:
        released_model_id (`str`):
            Hub id whose configuration is truncated.
        num_layers (`int`):
            Number of transformer layers to keep. The last kept layer must use full attention.

    Returns:
        `PreTrainedConfig`: The truncated configuration.
    """
    config = AutoConfig.from_pretrained(released_model_id)
    text_config = config.text_config
    layer_types = list(text_config.layer_types[:num_layers])
    if layer_types[-1] != "full_attention":
        raise ValueError(f"Layer {num_layers} of the released model is not a full-attention layer.")
    text_config.layer_types = layer_types
    text_config.num_hidden_layers = num_layers
    return config


BENCHMARK_PROMPTS = [
    "Explain how a hash map handles collisions and what that means for its worst-case lookup time.",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "Give a step-by-step plan for learning linear algebra in three months, with one resource per step.",
    "Write a Python function that merges overlapping intervals, then explain its time complexity.",
    "Summarize the causes and consequences of the Industrial Revolution for a high-school student.",
    "Compare electric and hydrogen fuel-cell cars in terms of efficiency, infrastructure and cost.",
    "Describe how vaccines train the immune system, and why some need booster doses.",
    "Draft a polite email asking a landlord to repair a broken heater before winter.",
]


def benchmark_prompts(args, family: str, model, device: torch.device):
    """
    Builds the prompts of a benchmark run.

    Args:
        args (`argparse.Namespace`):
            Parsed arguments providing `prompts`, `batch_size` and `prompt_length`.
        family (`str`):
            `"diffusion"` or `"gemma4"`, which selects the tokenizer for chat prompts.
        model (`PreTrainedModel`):
            Model the prompts are for.
        device (`torch.device`):
            Target device.

    Returns:
        `tuple[torch.Tensor, torch.Tensor | None]`: Token ids and the attention mask, which is `None` for random
        prompts.
    """
    if args.prompts == "random":
        vocab_size = model.config.text_config.vocab_size
        return random_prompts(args.batch_size, args.prompt_length, vocab_size, device), None
    tokenizer = AutoTokenizer.from_pretrained(DIFFUSION_MODEL_ID if family == "diffusion" else GEMMA4_MODEL_ID)
    tokenizer.padding_side = "left"
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": BENCHMARK_PROMPTS[index % len(BENCHMARK_PROMPTS)]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for index in range(args.batch_size)
    ]
    encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def random_prompts(batch_size: int, prompt_length: int, vocab_size: int, device: torch.device, seed: int = 0):
    """
    Draws random prompt token ids on the host and moves them to the device.

    Args:
        batch_size (`int`):
            Number of prompts.
        prompt_length (`int`):
            Tokens per prompt.
        vocab_size (`int`):
            Vocabulary size.
        device (`torch.device`):
            Target device.
        seed (`int`, *optional*, defaults to 0):
            Seed of the host generator.

    Returns:
        `torch.Tensor`: Token ids of shape `(batch_size, prompt_length)`.
    """
    generator = torch.Generator().manual_seed(seed)
    upper = min(vocab_size, 200_000)
    return torch.randint(1000 if upper > 2000 else 3, upper, (batch_size, prompt_length), generator=generator).to(
        device
    )


def environment_info(device: torch.device) -> dict:
    """
    Collects the software and hardware context of a benchmark run.

    Args:
        device (`torch.device`):
            Device the benchmark runs on.

    Returns:
        `dict`: Versions, device type and device count.
    """
    info = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "device": device.type,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import torch_tpu

        info["torch_tpu"] = getattr(torch_tpu, "__version__", "unknown")
    except ImportError:
        pass
    if device.type != "cpu":
        info["device_count"] = torch.accelerator.device_count()
    return info


def write_result(path: str | None, record: dict) -> None:
    """
    Prints a benchmark record and appends it to a JSON-lines file.

    Args:
        path (`str`, *optional*):
            Output file. Only prints when unset.
        record (`dict`):
            JSON-serializable record.
    """
    line = json.dumps(record)
    print(line, flush=True)
    if path is not None and os.environ.get("RANK", "0") == "0":
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a") as output_file:
            output_file.write(line + "\n")
