import argparse
import time

import torch
from common import DIFFUSION_MODEL_ID, DTYPES, environment_info, proxy_config, write_result
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaTextExperts

from dgemma_tpu.moe import DENSE_EXPERTS, DENSE_PREWEIGHTED_EXPERTS, register_experts_implementations
from dgemma_tpu.runtime import compile_backend_for, resolve_device, synchronize


def main() -> None:
    parser = argparse.ArgumentParser(description="Time one mixture-of-experts layer under each implementation.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--token-counts", type=int, nargs="+", default=[1, 8, 256, 1024, 2048])
    parser.add_argument(
        "--implementations",
        nargs="+",
        default=["grouped_mm", "batched_mm", DENSE_EXPERTS, DENSE_PREWEIGHTED_EXPERTS],
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    register_experts_implementations()
    device = resolve_device(args.device)
    dtype = DTYPES[args.dtype]
    text_config = proxy_config(DIFFUSION_MODEL_ID, 6).text_config
    experts = DiffusionGemmaTextExperts(text_config)
    torch.nn.init.normal_(experts.gate_up_proj, std=0.02)
    torch.nn.init.normal_(experts.down_proj, std=0.02)
    experts = experts.to(dtype).to(device).eval()
    backend = compile_backend_for(device, "auto")
    environment = environment_info(device)

    for num_tokens in args.token_counts:
        generator = torch.Generator().manual_seed(num_tokens)
        hidden_states = torch.randn(num_tokens, text_config.hidden_size, generator=generator).to(dtype).to(device)
        scores = torch.rand(num_tokens, text_config.num_experts, generator=generator)
        top_k_weights, top_k_index = torch.topk(scores, text_config.top_k_experts, dim=-1)
        top_k_weights = (top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)).to(device)
        top_k_index = top_k_index.to(device)
        reference = None
        for implementation in args.implementations:
            record = {
                "benchmark": "moe_formulations",
                "environment": environment,
                "implementation": implementation,
                "num_tokens": num_tokens,
            }
            try:
                text_config._experts_implementation = implementation
                torch._dynamo.reset()
                forward = torch.compile(experts.forward, backend=backend, fullgraph=True, dynamic=False)
                synchronize(device)
                start = time.perf_counter()
                output = forward(hidden_states, top_k_index, top_k_weights)
                synchronize(device)
                record["first_call_seconds"] = time.perf_counter() - start
                start = time.perf_counter()
                for _ in range(args.repeats):
                    output = forward(hidden_states, top_k_index, top_k_weights)
                synchronize(device)
                record["milliseconds"] = (time.perf_counter() - start) / args.repeats * 1000
                output = output.float().cpu()
                if reference is None:
                    reference = output
                record["max_abs_difference_to_first"] = float((output - reference).abs().max())
                record["reference_scale"] = float(reference.abs().mean())
                record["status"] = "ok"
            except Exception as error:
                record["status"] = "error"
                record["error"] = f"{type(error).__name__}: {str(error)[:300]}"
            write_result(args.output, record)


if __name__ == "__main__":
    with torch.no_grad():
        main()
