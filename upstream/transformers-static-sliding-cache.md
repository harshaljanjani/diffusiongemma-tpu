# DiffusionGemma: `cache_implementation="static"` decodes from a corrupted sliding-window cache once the context exceeds `sliding_window`

Status: draft, not filed. Target: huggingface/transformers. Checked on transformers 5.22.2, torch 2.15.0 (CPU).

## Summary

With `cache_implementation="static"`, `DiffusionGemmaForBlockDiffusion.generate` produces different tokens from the
default dynamic cache as soon as the number of cached tokens exceeds `sliding_window`. Below that length the two
caches agree token for token. For the released checkpoint (`sliding_window=1024`, `canvas_length=256`) this affects
any request where prompt plus generated tokens goes past 1024, which is the documented path for `torch.compile`.

## Cause

`DiffusionGemmaDecoderTextAttention.append_to_cache` places the canvas keys at
`cache_layer.cumulative_length` on the static path:

```python
cache_position_new = torch.arange(new_length, device=cache_layer.device) + cache_layer.cumulative_length
```

`StaticSlidingWindowLayer.update` only advances that tensor while the window is not yet full. Once the layer becomes
full it tracks the length in `cumulative_length_int` and leaves the tensor untouched ("we don't need to update it if
we are already outside of this branch"). The decoder therefore keeps writing the canvas at the stale position, on top
of cached tokens, and leaves zeros where the canvas is expected.

Observed on a 4-layer model with `sliding_window=24`, `canvas_length=16`, prompt length 12 (sliding layer 0, one row
per canvas):

| Canvas | Write position (`cumulative_length`) | True cached length (`cumulative_length_int`) |
|---|---|---|
| 1 | 12 | 12 |
| 2 | 12 | 28 |
| 3 | 12 | 44 |
| 4 | 12 | 60 |

## Reproduction

```python
import torch
from transformers import (
    DiffusionGemmaConfig,
    DiffusionGemmaForBlockDiffusion,
    DiffusionGemmaTextConfig,
    Gemma4VisionConfig,
)


def build(sliding_window):
    torch.manual_seed(0)
    text_config = DiffusionGemmaTextConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=sliding_window,
        max_position_embeddings=1024,
        layer_types=["sliding_attention", "full_attention"],
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=32,
    )
    vision_config = Gemma4VisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, image_size=32, patch_size=16
    )
    config = DiffusionGemmaConfig(
        text_config=text_config,
        vision_config=vision_config,
        canvas_length=16,
        boi_token_id=509,
        eoi_token_id=510,
        image_token_id=511,
    )
    return DiffusionGemmaForBlockDiffusion(config).eval()


def run(model, input_ids, cache_implementation):
    torch.manual_seed(1)
    kwargs = {"cache_implementation": cache_implementation} if cache_implementation else {}
    return model.generate(
        input_ids, max_new_tokens=48, max_denoising_steps=6, disable_compile=True, **kwargs
    ).sequences


input_ids = torch.randint(3, 500, (2, 12))
for sliding_window in (512, 24):
    model = build(sliding_window)
    same = torch.equal(run(model, input_ids, None), run(model, input_ids, "static"))
    print(f"sliding_window={sliding_window}: static == dynamic -> {same}")
```

Output:

```
sliding_window=512: static == dynamic -> True
sliding_window=24: static == dynamic -> False
```

In a sweep over prompt lengths 5, 12 and 40 and windows 8, 24 and 512, the first differing generated token always
falls in the first canvas decoded after the cached length passes the window. Models with only full-attention layers
never diverge.

## Related

- #49542 makes the sliding-window counter a tensor for autoregressive decode; it may fix this as a side effect.
- The same layer keeps its length in a Python int, which Dynamo guards on (`layers[0].cumulative_length_int == 28`),
  so the compiled encoder step recompiles once per canvas.

## How this repository avoids it

`dgemma_tpu.cache.BlockDiffusionStaticCache` keeps sliding layers right-aligned in a fixed window, appends with a
concatenate-and-trim, and tracks validity in device tensors. Its output is token-identical to the dynamic cache in
`tests/test_equivalence.py`, including past the window.
