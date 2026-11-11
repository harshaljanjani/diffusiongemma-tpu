import argparse
import time
import traceback

import torch
from common import environment_info, write_result

from dgemma_tpu.runtime import compile_backend_for, resolve_device, synchronize
from dgemma_tpu.sampler import entropy_bound_acceptance, finalize_canvas, gumbel_max_sample, token_distribution


def build_cases(device: torch.device, batch_size: int, canvas_length: int, vocab_size: int) -> dict:
    """
    Builds the operations the sampler and cache rely on, at the shapes of the released model.

    Args:
        device (`torch.device`):
            Device under test.
        batch_size (`int`):
            Batch size of the inputs.
        canvas_length (`int`):
            Canvas length of the inputs.
        vocab_size (`int`):
            Vocabulary size of the inputs.

    Returns:
        `dict[str, Callable]`: Zero-argument callables keyed by case name.
    """
    logits = torch.randn(batch_size, canvas_length, vocab_size, device=device)
    logits_bf16 = logits.to(torch.bfloat16)
    probs = torch.softmax(logits, dim=-1)
    entropy = torch.rand(batch_size, canvas_length, device=device)
    canvas = torch.randint(0, vocab_size, (batch_size, canvas_length), device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    eos_token_ids = torch.tensor([1, 106, 50], device=device)
    keys = torch.zeros(batch_size, 8, 1024, 256, device=device, dtype=torch.bfloat16)
    new_keys = torch.ones(batch_size, 8, canvas_length, 256, device=device, dtype=torch.bfloat16)
    length = torch.tensor(128, device=device)
    embedding = torch.randn(vocab_size, 64, device=device, dtype=torch.bfloat16)

    return {
        "rand_uniform": lambda: torch.rand(batch_size, canvas_length, vocab_size, device=device),
        "randint": lambda: torch.randint(0, vocab_size, (batch_size, canvas_length), device=device),
        "log_softmax": lambda: torch.log_softmax(logits, dim=-1),
        "token_distribution": lambda: token_distribution(logits),
        "token_distribution_bf16_input": lambda: token_distribution(logits_bf16),
        "categorical_entropy": lambda: torch.distributions.Categorical(logits=logits).entropy(),
        "gumbel_max_sample": lambda: gumbel_max_sample(
            torch.log_softmax(logits, dim=-1), torch.rand(logits.shape, device=device)
        ),
        "multinomial_float32": lambda: torch.multinomial(probs.view(-1, vocab_size), num_samples=1),
        "multinomial_bf16": lambda: torch.multinomial(probs.view(-1, vocab_size).to(torch.bfloat16), num_samples=1),
        "argmax_vocab": lambda: torch.argmax(logits, dim=-1),
        "sort_canvas": lambda: torch.sort(entropy, dim=-1),
        "cumsum_canvas": lambda: torch.cumsum(entropy, dim=-1),
        "scatter_bool": lambda: torch.scatter(
            torch.zeros_like(entropy, dtype=torch.bool), -1, torch.argsort(entropy, dim=-1), entropy > 0.5
        ),
        "entropy_bound_acceptance": lambda: entropy_bound_acceptance(entropy, 0.1),
        "isin": lambda: torch.isin(canvas, eos_token_ids),
        "finalize_canvas": lambda: finalize_canvas(canvas, finished, eos_token_ids, 0),
        "index_copy_tensor_position": lambda: keys.index_copy(
            2, torch.arange(canvas_length, device=device) + length, new_keys
        ),
        "index_copy_inplace": lambda: keys.index_copy_(
            2, torch.arange(canvas_length, device=device) + length, new_keys
        ),
        "cat_window": lambda: torch.cat([keys, new_keys], dim=2)[:, :, -1024:],
        "copy_inplace": lambda: keys.copy_(torch.cat([keys, new_keys], dim=2)[:, :, -1024:]),
        "zero_dim_add_inplace": lambda: length.add_(0),
        "soft_embedding_matmul": lambda: torch.matmul(probs.to(torch.bfloat16), embedding),
        "where_3d": lambda: torch.where(finished[:, None, None], logits, logits * 2),
        "history_compare": lambda: (canvas[None] == canvas[None]).all(dim=-1).all(dim=0),
        "bool_sync": lambda: bool(finished.all()),
        "tanh_softcap": lambda: torch.tanh(logits / 30.0) * 30.0,
    }


def run_case(name: str, function, device: torch.device, backend: str | None, repeats: int) -> dict:
    """
    Runs one case eagerly or compiled and records whether it works and how long it takes.

    Args:
        name (`str`):
            Case name.
        function (`Callable`):
            Zero-argument callable.
        device (`torch.device`):
            Device under test.
        backend (`str`, *optional*):
            `torch.compile` backend, or `None` for eager execution.
        repeats (`int`):
            Number of timed repetitions after the first call.

    Returns:
        `dict`: Case name, mode, status, first-call seconds and steady-state seconds.
    """
    mode = "eager" if backend is None else f"compile:{backend}"
    record = {"case": name, "mode": mode}
    try:
        callable_under_test = function if backend is None else torch.compile(function, backend=backend, dynamic=False)
        synchronize(device)
        start = time.perf_counter()
        callable_under_test()
        synchronize(device)
        record["first_call_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(repeats):
            callable_under_test()
        synchronize(device)
        record["steady_seconds"] = (time.perf_counter() - start) / max(repeats, 1)
        record["status"] = "ok"
    except Exception as error:
        record["status"] = "error"
        record["error"] = f"{type(error).__name__}: {str(error)[:500]}"
        record["traceback_tail"] = traceback.format_exc()[-800:]
    return record


def check_random_streams(device: torch.device, backend: str | None) -> dict:
    """
    Checks that compiled random draws change between calls and repeat after reseeding.

    Args:
        device (`torch.device`):
            Device under test.
        backend (`str`, *optional*):
            `torch.compile` backend, or `None` for eager execution.

    Returns:
        `dict`: Whether consecutive draws differ and whether reseeding reproduces them.
    """

    def draw():
        return torch.rand(4, 8, device=device), torch.randint(0, 1000, (4, 8), device=device)

    record = {"case": "random_streams", "mode": "eager" if backend is None else f"compile:{backend}"}
    try:
        sample = draw if backend is None else torch.compile(draw, backend=backend, dynamic=False)
        torch.manual_seed(0)
        first = [tensor.cpu() for tensor in sample()]
        second = [tensor.cpu() for tensor in sample()]
        torch.manual_seed(0)
        repeated = [tensor.cpu() for tensor in sample()]
        record["consecutive_draws_differ"] = not torch.equal(first[0], second[0])
        record["reseed_reproduces_uniform"] = torch.equal(first[0], repeated[0])
        record["reseed_reproduces_tokens"] = torch.equal(first[1], repeated[1])
        record["status"] = "ok"
    except Exception as error:
        record["status"] = "error"
        record["error"] = f"{type(error).__name__}: {str(error)[:500]}"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the operations used by the static sampler on a device.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--canvas-length", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=262144)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    backends = [None, compile_backend_for(device, "auto")]
    environment = environment_info(device)
    for backend in backends:
        write_result(
            args.output,
            {"benchmark": "op_coverage", "environment": environment, **check_random_streams(device, backend)},
        )
        cases = build_cases(device, args.batch_size, args.canvas_length, args.vocab_size)
        for name, function in cases.items():
            record = run_case(name, function, device, backend, args.repeats)
            write_result(args.output, {"benchmark": "op_coverage", "environment": environment, **record})


if __name__ == "__main__":
    with torch.no_grad():
        main()
