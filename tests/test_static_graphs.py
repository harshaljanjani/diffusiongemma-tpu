import pytest
import torch

from dgemma_tpu import StaticBlockDiffusionGenerator
from dgemma_tpu.instrument import GraphCountingBackend, HostSyncCounter

GENERATION_KWARGS = {"max_denoising_steps": 8, "eos_token_id": 1, "pad_token_id": 0}


def make_prompt(seed, batch_size, prompt_length):
    torch.manual_seed(seed)
    return torch.randint(3, 500, (batch_size, prompt_length))


@pytest.fixture
def reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def test_each_stage_compiles_once(tiny_model_factory, reset_dynamo):
    model = tiny_model_factory(sliding_window=24)
    backend = GraphCountingBackend()
    generator = StaticBlockDiffusionGenerator(
        model, max_new_tokens=64, prompt_bucket=16, stop_check_interval=4, compile_backend=backend, **GENERATION_KWARGS
    )
    generator.generate(make_prompt(seed=1, batch_size=2, prompt_length=13))
    assert backend.num_graphs == 4

    generator.generate(make_prompt(seed=2, batch_size=2, prompt_length=9))
    generator.generate(make_prompt(seed=3, batch_size=2, prompt_length=16))
    assert backend.num_graphs == 4


def test_new_prompt_bucket_only_recompiles_shape_dependent_stages(tiny_model_factory, reset_dynamo):
    model = tiny_model_factory(sliding_window=24)
    backend = GraphCountingBackend()
    generator = StaticBlockDiffusionGenerator(
        model, max_new_tokens=32, prompt_bucket=16, compile_backend=backend, **GENERATION_KWARGS
    )
    generator.generate(make_prompt(seed=1, batch_size=1, prompt_length=10))
    graphs_after_first_bucket = backend.num_graphs
    generator.generate(make_prompt(seed=1, batch_size=1, prompt_length=20))
    assert backend.num_graphs <= 2 * graphs_after_first_bucket
    generator.generate(make_prompt(seed=2, batch_size=1, prompt_length=30))
    generator.generate(make_prompt(seed=3, batch_size=1, prompt_length=7))
    assert backend.num_graphs <= 2 * graphs_after_first_bucket


def test_compiled_matches_eager(tiny_model_factory, reset_dynamo):
    model = tiny_model_factory(sliding_window=24)
    input_ids = make_prompt(seed=1, batch_size=2, prompt_length=13)
    outputs = []
    for compile_backend in (None, "aot_eager"):
        generator = StaticBlockDiffusionGenerator(
            model, max_new_tokens=64, prompt_bucket=16, compile_backend=compile_backend, **GENERATION_KWARGS
        )
        torch.manual_seed(11)
        outputs.append(generator.generate(input_ids))
    assert torch.equal(outputs[0].sequences, outputs[1].sequences)
    assert torch.equal(outputs[0].decoder_forward_passes, outputs[1].decoder_forward_passes)


@pytest.mark.parametrize("stop_check_interval", [0, 1, 4])
def test_host_syncs_are_bounded_by_stop_checks(tiny_model_factory, stop_check_interval):
    model = tiny_model_factory(sliding_window=24)
    generator = StaticBlockDiffusionGenerator(
        model, max_new_tokens=64, prompt_bucket=16, stop_check_interval=stop_check_interval, **GENERATION_KWARGS
    )
    input_ids = make_prompt(seed=1, batch_size=2, prompt_length=13)
    with HostSyncCounter() as counter:
        output = generator.generate(input_ids)
    assert counter.count <= output.num_stop_checks + output.num_canvases
    if stop_check_interval == 0:
        assert output.num_stop_checks == 0
        assert output.num_denoising_steps == output.num_canvases * 8


def test_stop_check_interval_does_not_change_first_canvas(tiny_model_factory):
    model = tiny_model_factory(sliding_window=24)
    input_ids = make_prompt(seed=1, batch_size=2, prompt_length=13)
    first_canvases = []
    for stop_check_interval in (1, 3, 0):
        generator = StaticBlockDiffusionGenerator(
            model, max_new_tokens=16, prompt_bucket=16, stop_check_interval=stop_check_interval, **GENERATION_KWARGS
        )
        torch.manual_seed(11)
        first_canvases.append(generator.generate(input_ids).sequences)
    assert torch.equal(first_canvases[0], first_canvases[1])
    assert torch.equal(first_canvases[0], first_canvases[2])
