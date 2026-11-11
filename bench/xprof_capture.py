import argparse

import torch
from common import DTYPES, load_model, random_prompts

from dgemma_tpu import StaticBlockDiffusionGenerator
from dgemma_tpu.runtime import compile_backend_for, resolve_device, synchronize


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture an XProf trace of compiled block-diffusion decoding.")
    parser.add_argument("--model", default="proxy:6")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-denoising-steps", type=int, default=12)
    parser.add_argument("--output-dir", default="results/xprof")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model = load_model("diffusion", args.model, device, DTYPES[args.dtype])
    generator = StaticBlockDiffusionGenerator(
        model,
        max_new_tokens=args.max_new_tokens,
        stop_check_interval=0,
        compile_backend=compile_backend_for(device, "auto"),
        max_denoising_steps=args.max_denoising_steps,
        pad_token_id=0,
    )
    input_ids = random_prompts(args.batch_size, args.prompt_length, model.config.text_config.vocab_size, device)
    for _ in range(2):
        generator.generate(input_ids)
    synchronize(device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    profiler_kwargs = {}
    if device.type == "tpu":
        from torch.tpu.profiler import TpuProfilerConfig

        activities.append(torch.profiler.ProfilerActivity.TPU)
        profiler_kwargs["experimental_config"] = TpuProfilerConfig()
    with torch.profiler.profile(
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.output_dir),
        **profiler_kwargs,
    ) as profiler:
        generator.generate(input_ids)
        synchronize(device)
        profiler.step()
    print(f"trace written to {args.output_dir}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
