from contextlib import contextmanager

import torch

from dgemma_tpu.sampler import gumbel_max_sample


@contextmanager
def gumbel_multinomial():
    """Replaces `torch.multinomial` with Gumbel-max sampling drawn from the default generator."""
    original = torch.multinomial

    def multinomial(probs, num_samples=1, **kwargs):
        uniform = torch.rand(probs.shape, dtype=torch.float32)
        return gumbel_max_sample(torch.log(probs), uniform).unsqueeze(-1)

    torch.multinomial = multinomial
    try:
        yield
    finally:
        torch.multinomial = original


def reference_generate(model, input_ids, seed, attention_mask=None, **generation_kwargs):
    """Runs the Transformers sampler with a dynamic cache and Gumbel-max sampling."""
    if attention_mask is not None:
        generation_kwargs["attention_mask"] = attention_mask
    torch.manual_seed(seed)
    with gumbel_multinomial():
        return model.generate(input_ids, disable_compile=True, **generation_kwargs)
