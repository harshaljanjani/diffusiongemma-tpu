import argparse
import re

import torch
from common import DIFFUSION_MODEL_ID, DTYPES, GEMMA4_MODEL_ID, environment_info, load_model, write_result
from datasets import load_dataset
from transformers import AutoProcessor

from dgemma_tpu import StaticBlockDiffusionGenerator
from dgemma_tpu.ar_baseline import StaticAutoregressiveGenerator
from dgemma_tpu.runtime import compile_backend_for, resolve_device

INSTRUCTION = "Solve the problem. Finish with a line of the form 'Answer: <number>'.\n\n"
NUMBER_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_number(text: str) -> float | None:
    """
    Extracts the final numeric answer from a model response or a GSM8K reference solution.

    Args:
        text (`str`):
            Text containing the answer.

    Returns:
        `float` or `None`: The number after the last answer marker, or the last number in the text.
    """
    has_marker = False
    for marker in ("####", "Answer:"):
        if marker in text:
            text = text.rsplit(marker, 1)[1]
            has_marker = True
            break
    matches = NUMBER_PATTERN.findall(text)
    if not matches:
        return None
    candidate = matches[0] if has_marker else matches[-1]
    try:
        return float(candidate.replace(",", "").rstrip("."))
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a GSM8K subset with static-shape decoding.")
    parser.add_argument("--family", required=True, choices=["diffusion", "gemma4"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--compile", default="auto")
    parser.add_argument("--num-examples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--prompt-bucket", type=int, default=256)
    parser.add_argument("--stop-check-interval", type=int, default=4)
    parser.add_argument("--tensor-parallel", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    model_id = args.model or (DIFFUSION_MODEL_ID if args.family == "diffusion" else GEMMA4_MODEL_ID)
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    model = load_model(args.family, model_id, device, DTYPES[args.dtype], tensor_parallel=args.tensor_parallel)

    backend = compile_backend_for(device, args.compile)
    if args.family == "diffusion":
        generator = StaticBlockDiffusionGenerator(
            model,
            max_new_tokens=args.max_new_tokens,
            prompt_bucket=args.prompt_bucket,
            stop_check_interval=args.stop_check_interval,
            compile_backend=backend,
        )
    else:
        generator = StaticAutoregressiveGenerator(
            model, max_new_tokens=args.max_new_tokens, prompt_bucket=args.prompt_bucket, compile_backend=backend
        )

    dataset = load_dataset("openai/gsm8k", "main", split="test").select(range(args.num_examples))
    torch.manual_seed(args.seed)
    num_correct = 0
    examples = []
    for start in range(0, len(dataset), args.batch_size):
        batch = dataset[start : start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": INSTRUCTION + question}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for question in batch["question"]
        ]
        while len(prompts) < args.batch_size:
            prompts.append(prompts[-1])
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        output = generator.generate(encoded["input_ids"], encoded["attention_mask"])
        completions = tokenizer.batch_decode(
            output.sequences[:, encoded["input_ids"].shape[1] :].cpu(), skip_special_tokens=True
        )
        for question, reference, completion in zip(batch["question"], batch["answer"], completions):
            predicted = extract_number(completion)
            expected = extract_number(reference)
            is_correct = predicted is not None and expected is not None and abs(predicted - expected) < 1e-4
            num_correct += int(is_correct)
            examples.append(
                {
                    "question": question[:200],
                    "completion": completion[-400:],
                    "expected": expected,
                    "correct": is_correct,
                }
            )

    record = {
        "benchmark": "gsm8k",
        "args": vars(args),
        "environment": environment_info(device),
        "accuracy": num_correct / len(dataset),
        "num_examples": len(dataset),
        "examples": examples,
    }
    write_result(args.output, record)


if __name__ == "__main__":
    with torch.no_grad():
        main()
