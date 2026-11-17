import pytest
import torch
from reference import reference_generate, reference_numerics, reference_token_distribution

from dgemma_tpu import StaticBlockDiffusionGenerator
from dgemma_tpu.sampler import token_distribution


def static_generate(model, input_ids, seed, max_new_tokens, **kwargs):
    generator = StaticBlockDiffusionGenerator(model, max_new_tokens=max_new_tokens, stop_check_interval=1, **kwargs)
    torch.manual_seed(seed)
    with reference_numerics():
        return generator.generate(input_ids)


@pytest.mark.parametrize("sliding_window", [8, 24, 512])
@pytest.mark.parametrize("prompt_length", [5, 12, 40])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_matches_transformers_dynamic_cache(tiny_model_factory, sliding_window, prompt_length, batch_size):
    model = tiny_model_factory(sliding_window=sliding_window)
    torch.manual_seed(7)
    input_ids = torch.randint(3, 500, (batch_size, prompt_length))
    reference = reference_generate(
        model, input_ids, seed=3, max_new_tokens=64, max_denoising_steps=8, eos_token_id=1, pad_token_id=0
    )
    output = static_generate(
        model,
        input_ids,
        seed=3,
        max_new_tokens=64,
        prompt_bucket=1,
        max_denoising_steps=8,
        eos_token_id=1,
        pad_token_id=0,
    )
    assert torch.equal(output.sequences, reference.sequences)
    assert torch.allclose(output.tokens_per_forward, reference.tokens_per_forward)


@pytest.mark.parametrize("prompt_bucket", [16, 64])
def test_prompt_bucket_padding_does_not_change_tokens(tiny_model_factory, prompt_bucket):
    model = tiny_model_factory(sliding_window=24)
    torch.manual_seed(7)
    input_ids = torch.randint(3, 500, (2, 13))
    reference = reference_generate(
        model, input_ids, seed=5, max_new_tokens=48, max_denoising_steps=8, eos_token_id=1, pad_token_id=0
    )
    output = static_generate(
        model,
        input_ids,
        seed=5,
        max_new_tokens=48,
        prompt_bucket=prompt_bucket,
        max_denoising_steps=8,
        eos_token_id=1,
        pad_token_id=0,
    )
    assert torch.equal(output.sequences, reference.sequences)


def test_token_distribution_matches_reference_numerics():
    torch.manual_seed(0)
    logits = torch.randn(2, 16, 512) * 4
    log_probs, probs, entropy = token_distribution(logits)
    reference_log_probs, reference_probs, reference_entropy = reference_token_distribution(logits)
    assert torch.allclose(log_probs, reference_log_probs, atol=1e-5)
    assert torch.allclose(probs, reference_probs, atol=1e-6)
    assert torch.allclose(entropy, reference_entropy, atol=1e-5)
