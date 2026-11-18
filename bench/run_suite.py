import argparse
import copy
import gc

import quality_gsm8k
import run_ar
import run_diffusion
import torch
from common import DIFFUSION_MODEL_ID, DTYPES, GEMMA4_MODEL_ID, load_model, write_result
from transformers import AutoTokenizer

from dgemma_tpu.moe import TOKEN_COUNT_EXPERTS, set_experts_implementation
from dgemma_tpu.runtime import resolve_device, timed

VARIANTS = {
    "grouped": {"experts_implementation": "grouped_mm"},
    "check1": {"stop_check_interval": 1},
    "check0": {"stop_check_interval": 0},
    "sampled": {"do_sample": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load one model and benchmark several configurations in-process.")
    parser.add_argument("--family", required=True, choices=["diffusion", "gemma4"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--compile", default="auto")
    parser.add_argument("--expert-parallel", action="store_true")
    parser.add_argument("--experts-implementation", default=TOKEN_COUNT_EXPERTS)
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=[1])
    parser.add_argument("--variant-batch-size", type=int, default=1)
    parser.add_argument("--variants", nargs="*", default=[], choices=sorted(VARIANTS))
    parser.add_argument("--prompts", default="chat", choices=["random", "chat"])
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--prompt-bucket", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--stop-check-interval", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--timed-runs", type=int, default=3)
    parser.add_argument("--gsm8k-examples", type=int, default=0)
    parser.add_argument("--gsm8k-batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default="")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def configuration(args: argparse.Namespace, batch_size: int, variant: str | None) -> argparse.Namespace:
    """
    Builds the single-run arguments of one configuration of the suite.

    Args:
        args (`argparse.Namespace`):
            Suite arguments.
        batch_size (`int`):
            Batch size of the configuration.
        variant (`str`, *optional*):
            Name of a variant in `VARIANTS` to apply on top of the defaults.

    Returns:
        `argparse.Namespace`: Arguments accepted by the family's `benchmark` function.
    """
    config = copy.copy(args)
    config.batch_size = batch_size
    config.sampler = "static"
    config.cache = "static"
    config.max_denoising_steps = None
    config.attn_implementation = None
    config.do_sample = False
    if config.stop_check_interval is None:
        config.stop_check_interval = 4 if args.family == "diffusion" else 16
    for name, value in VARIANTS.get(variant, {}).items():
        setattr(config, name, value)
    config.label = f"{args.label}{args.family}_b{batch_size}" + (f"_{variant}" if variant else "")
    return config


def release_compiled_state(device: torch.device) -> None:
    """
    Drops compiled graphs and cached generation buffers between configurations.

    Args:
        device (`torch.device`):
            Device whose allocator cache is emptied when the backend supports it.
    """
    torch._dynamo.reset()
    gc.collect()
    if device.type != "cpu" and hasattr(torch.accelerator, "empty_cache"):
        torch.accelerator.empty_cache()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model_id = args.model or (DIFFUSION_MODEL_ID if args.family == "diffusion" else GEMMA4_MODEL_ID)
    args.model = model_id
    timings = {}
    with timed(device, timings, "load_seconds"):
        model = load_model(
            args.family, model_id, device, DTYPES[args.dtype], None, args.experts_implementation, args.expert_parallel
        )
    benchmark = run_diffusion.benchmark if args.family == "diffusion" else run_ar.benchmark

    configurations = [(batch_size, None) for batch_size in args.batch_sizes]
    configurations += [(args.variant_batch_size, variant) for variant in args.variants]
    for batch_size, variant in configurations:
        config = configuration(args, batch_size, variant)
        set_experts_implementation(model, config.experts_implementation)
        try:
            record = benchmark(config, model, device, timings["load_seconds"])
        except Exception as error:
            record = {
                "benchmark": "error",
                "args": vars(config),
                "error": f"{type(error).__name__}: {str(error)[:600]}",
            }
        write_result(args.output, record)
        release_compiled_state(device)

    if args.gsm8k_examples > 0:
        set_experts_implementation(model, args.experts_implementation)
        quality_args = configuration(args, args.gsm8k_batch_size, None)
        quality_args.num_examples = args.gsm8k_examples
        quality_args.label = f"{args.label}{args.family}_gsm8k"
        released_id = DIFFUSION_MODEL_ID if args.family == "diffusion" else GEMMA4_MODEL_ID
        tokenizer = AutoTokenizer.from_pretrained(released_id)
        tokenizer.padding_side = "left"
        write_result(args.output, quality_gsm8k.evaluate(quality_args, model, tokenizer, device))


if __name__ == "__main__":
    with torch.no_grad():
        main()
