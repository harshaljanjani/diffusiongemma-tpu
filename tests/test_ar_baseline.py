import pytest
import torch

from dgemma_tpu.ar_baseline import StaticAutoregressiveGenerator
from dgemma_tpu.instrument import GraphCountingBackend


def greedy_reference(model, input_ids, max_new_tokens):
    return model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)


@pytest.mark.parametrize("sliding_window", [8, 24, 512])
@pytest.mark.parametrize("prompt_bucket", [1, 16])
def test_greedy_matches_transformers(tiny_gemma4_factory, sliding_window, prompt_bucket):
    model = tiny_gemma4_factory(sliding_window=sliding_window)
    torch.manual_seed(7)
    input_ids = torch.randint(3, 500, (2, 13))
    reference = greedy_reference(model, input_ids, max_new_tokens=40)
    generator = StaticAutoregressiveGenerator(
        model, max_new_tokens=40, prompt_bucket=prompt_bucket, stop_check_interval=1
    )
    output = generator.generate(input_ids)
    common_length = min(output.sequences.shape[1], reference.shape[1])
    assert common_length > input_ids.shape[1] + 8
    assert torch.equal(output.sequences[:, :common_length], reference[:, :common_length])


def test_decode_step_compiles_once(tiny_gemma4_factory):
    torch._dynamo.reset()
    model = tiny_gemma4_factory()
    backend = GraphCountingBackend()
    generator = StaticAutoregressiveGenerator(
        model, max_new_tokens=40, prompt_bucket=16, stop_check_interval=0, compile_backend=backend
    )
    torch.manual_seed(7)
    generator.generate(torch.randint(3, 500, (2, 13)))
    generator.generate(torch.randint(3, 500, (2, 9)))
    assert backend.num_graphs == 2
    torch._dynamo.reset()
