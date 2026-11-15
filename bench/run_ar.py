import argparse
import statistics
import time

import torch
from common import DTYPES, benchmark_prompts, environment_info, load_model, write_result

from dgemma_tpu.ar_baseline import StaticAutoregressiveGenerator
from dgemma_tpu.dist import spmd_decorator
from dgemma_tpu.runtime import compile_backend_for, peak_memory_bytes, resolve_device, synchronize, timed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark autoregressive decoding of Gemma 4.")
    parser.add_argument("--model", default="proxy:6", help="proxy:<num_layers>, or a pretrained id or path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--sampler", default="static", choices=["static", "transformers"])
    parser.add_argument("--compile", default="auto", help="none, auto, or a torch.compile backend name")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompts", default="random", choices=["random", "chat"])
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--prompt-bucket", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--stop-check-interval", type=int, default=16)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--experts-implementation", default=None)
    parser.add_argument("--expert-parallel", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--timed-runs", type=int, default=3)
    parser.add_argument("--label", default="")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def run_static(args, model, device, input_ids, attention_mask):
    generator = StaticAutoregressiveGenerator(
        model,
        max_new_tokens=args.max_new_tokens,
        prompt_bucket=args.prompt_bucket,
        stop_check_interval=args.stop_check_interval,
        do_sample=args.do_sample,
        compile_backend=compile_backend_for(device, args.compile),
        function_decorator=spmd_decorator(device.type) if args.expert_parallel else None,
    )

    def generate():
        output = generator.generate(input_ids, attention_mask, profile=True)
        return {
            "sequences": output.sequences,
            "num_decode_steps": output.num_decode_steps,
            "num_stop_checks": output.num_stop_checks,
            "prefill_seconds": output.prefill_seconds,
        }

    return generate


def run_transformers(args, model, device, input_ids, attention_mask):
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "cache_implementation": "static",
    }
    if args.compile == "none":
        kwargs["disable_compile"] = True
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask

    def generate():
        return {"sequences": model.generate(input_ids, **kwargs)}

    return generate


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    timings = {}
    with timed(device, timings, "load_seconds"):
        model = load_model(
            "gemma4",
            args.model,
            device,
            DTYPES[args.dtype],
            args.attn_implementation,
            args.experts_implementation,
            args.expert_parallel,
        )
    input_ids, attention_mask = benchmark_prompts(args, "gemma4", model, device)
    prompt_length = input_ids.shape[1]
    build = run_static if args.sampler == "static" else run_transformers
    generate = build(args, model, device, input_ids, attention_mask)

    warmup_seconds = []
    for _ in range(args.warmup_runs):
        synchronize(device)
        start = time.perf_counter()
        generate()
        synchronize(device)
        warmup_seconds.append(time.perf_counter() - start)

    runs = []
    for _ in range(args.timed_runs):
        synchronize(device)
        start = time.perf_counter()
        result = generate()
        synchronize(device)
        elapsed = time.perf_counter() - start
        sequences = result.pop("sequences")
        new_tokens = sequences[:, prompt_length:]
        decode_seconds = elapsed - (result.get("prefill_seconds") or 0.0)
        result.update(
            total_seconds=elapsed,
            generated_tokens=int((new_tokens != 0).sum()),
            decoded_tokens=new_tokens.numel(),
            tokens_per_second=int((new_tokens != 0).sum()) / decode_seconds,
            decoded_tokens_per_second=new_tokens.numel() / decode_seconds,
            seconds_per_step=decode_seconds / new_tokens.shape[1],
        )
        runs.append(result)

    record = {
        "benchmark": "autoregressive",
        "args": vars(args),
        "environment": environment_info(device),
        "load_seconds": timings["load_seconds"],
        "warmup_seconds": warmup_seconds,
        "median_tokens_per_second": statistics.median(run["tokens_per_second"] for run in runs),
        "median_total_seconds": statistics.median(run["total_seconds"] for run in runs),
        "peak_memory_bytes": peak_memory_bytes(device),
        "runs": runs,
    }
    write_result(args.output, record)


if __name__ == "__main__":
    with torch.no_grad():
        main()
