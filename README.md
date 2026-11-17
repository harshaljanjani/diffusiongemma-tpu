# DiffusionGemma on TorchTPU

Static-shape block-diffusion decoding for [DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
in native PyTorch on Cloud TPU v6e with [TorchTPU](https://github.com/google-pytorch/torch_tpu), and a like-for-like
benchmark against autoregressive [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it), the model it
was fine-tuned from.

Built for the TorchTPU Content Sprint 2026.

## What is in the repository

| Path | Contents |
|---|---|
| `dgemma_tpu/generate.py` | `StaticBlockDiffusionGenerator`: the released entropy-bound sampler with fixed shapes, one compiled graph per stage and a host sync only every few steps |
| `dgemma_tpu/cache.py` | KV cache whose length and validity live in device tensors, including for sliding-window layers |
| `dgemma_tpu/ar_baseline.py` | `StaticAutoregressiveGenerator` for Gemma 4, built on the same cache and stopping policy |
| `dgemma_tpu/moe.py` | Dense mixture-of-experts implementations and a selector by token count |
| `dgemma_tpu/dist.py` | Multi-chip execution that slices every expert across chips and all-reduces once per layer |
| `bench/` | Operator coverage, per-layer expert timings, single-run and suite benchmarks, GSM8K check, XProf capture |
| `cloud/` | Scripts that create a Spot TPU VM, run each session and record its cost |
| `results/` | Raw JSON-lines output and logs of every session |
| `upstream/` | Drafted reports for Transformers and torch_tpu |
| `tests/` | CPU tests, including token-for-token equality with the Transformers sampler |

## Why the Transformers sampler needs rewriting for TPU

`DiffusionGemmaForBlockDiffusion.generate` in Transformers 5.22.2 was written around CUDA graphs. On TPU:

| Pattern in the Transformers sampler | Effect on TPU | What this repository does |
|---|---|---|
| `torch.compile(mode="reduce-overhead")` for every compiled function | The TPU backend rejects the mode, so the compiled path does not run at all | Compiles four stage functions with `backend="tpu", dynamic=False` |
| `if torch.all(finished_denoising): break` after every step | One host sync per step | Checks every `stop_check_interval` steps; rows that finished are frozen on device |
| Self-conditioning logits are `None` on the first step | Two graphs for one step function | Zero soft embeddings on the first step |
| `(batch, 256, 262144)` logits carried between steps, softmax recomputed in the decoder | Large tensor crosses the step boundary; four normalizations per step | Carries `(batch, 256, 2816)` soft embeddings; one `log_softmax` feeds sampling, entropy and self-conditioning |
| `torch.cat` grows the sequence and masks per canvas; boolean-mask assignment when finalizing | New shapes per canvas | Preallocated buffers, `index_copy_` at a device-side length, `torch.where` |
| Static sliding-window cache keeps its length in a Python int | Guard on the int recompiles the encoder per canvas; decoder reads a corrupted window past 1024 tokens | Right-aligned window buffer with validity tracked in tensors |
| Default `grouped_mm` experts | About 3 ms per layer regardless of token count | Implementation chosen by token count |
| `model.to("tpu")` | Encoder and decoder weights become two copies | Each distinct parameter is moved once |

On CPU the rewritten sampler produces the same tokens as the Transformers dynamic-cache sampler when both draw from
the same noise (`tests/test_equivalence.py`), compiles each stage exactly once and syncs with the host only at stop
checks (`tests/test_static_graphs.py`).

## Single-chip results (TPU v6e, one chip)

These runs use a randomly initialized model with the released width, vocabulary, expert count and canvas length but
6 of the 30 layers, so that it fits one 32 GB chip. They isolate per-step costs; they say nothing about text quality.
Prompt length 200, bf16, 512 new tokens for diffusion and 256 for autoregressive decoding.

### Execution modes, 256-token denoising step, batch 1

| Mode | Time per step |
|---|---|
| Strict eager | 92.2 ms |
| DeferAndFuse (`TORCH_TPU_DEFER_AND_FUSE=1`) | 40.7 ms |
| `torch.compile(backend="tpu")`, default experts | 26.2 ms |
| `torch.compile(backend="tpu")`, experts chosen by token count | 15.9 ms |
| Transformers sampler, dynamic cache, uncompiled | 9.11 s per 512 tokens, against 1.40 s for the row above |
| Transformers sampler, static cache, compiled | fails: `mode='reduce-overhead'` is not accepted |

Checking for early stopping after every step instead of never changes the step time from 25.7 ms to 27.0 ms.

### One mixture-of-experts layer

128 experts, hidden size 2816, intermediate size 704, eight experts per token, compiled, milliseconds:

| Tokens in the pass | `grouped_mm` (Transformers default) | `batched_mm` | Dense, all experts (this repository) |
|---|---|---|---|
| 1 | 3.10 | 0.13 | 1.31 |
| 8 | 3.08 | 1.92 | 1.32 |
| 256 | 3.44 | 64.98 | 1.38 |
| 1024 | 3.74 | compile-time out of memory | 2.37 |
| 2048 | 5.27 | compile-time out of memory | 4.96 |

The default costs about 3 ms whether it processes one token or 256. An XProf trace
(`results/session1/xprof_denoise_proxy6_grouped_mm.trace.json.gz`) shows a copy of the full
`bf16[128,1408,2816]` expert tensor and offloaded gathers on every step. `dgemma_tpu.moe` picks `batched_mm` up to
4 tokens, the dense form up to 1536 tokens and `grouped_mm` beyond.

### Effect on both decoders, batch 1

| Decoder | Default experts | Experts by token count | Speedup |
|---|---|---|---|
| Diffusion, time per 256-token step | 26.6 ms | 15.9 ms | 1.7x |
| Autoregressive, time per token | 20.8 ms | 2.7 ms | 7.7x |

The autoregressive baseline gains far more than the diffusion decoder from the same fix. With the default experts a
single-token step costs almost as much as a 256-token step, which would have flattered diffusion by a wide margin.

### Depth and batch scaling

| Model depth | Diffusion step, batch 1 | Autoregressive token, batch 1 |
|---|---|---|
| 6 layers | 15.9 ms | 2.68 ms |
| 12 layers | 27.1 ms | 4.00 ms |
| Per additional layer | 1.87 ms | 0.22 ms |

| Batch | Diffusion step (6 layers) | Autoregressive token (6 layers) |
|---|---|---|
| 1 | 15.9 ms | 2.69 ms |
| 2 | 22.3 ms | 5.29 ms |
| 4 | 33.5 ms | 8.40 ms |
| 8 | 61.0 ms | 12.3 ms |
| 16 | out of memory before the tie fix | 15.1 ms |

## Reproducing

Local tests need only CPU PyTorch:

```
uv venv --python 3.12 .venv
uv pip install --python .venv "torch==2.15.0" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv "transformers==5.22.2" datasets pytest
.venv/bin/python -m pytest
```

On a TPU VM created with `cloud/create_vm.sh` (its startup script installs `torch_tpu==0.2.0`):

```
bash cloud/run_session1.sh     # one chip: operator coverage, execution modes, proxy models
bash cloud/run_session1b.sh    # one chip: expert implementations for both decoders
bash cloud/run_session2.sh     # four chips: released weights for both models
```

Costs are in `cloud/ledger.md`.

## Upstream

`upstream/findings.md` lists six issues found along the way in Transformers and torch_tpu, each with a reproduction
or a pointer to the raw log. None has been filed yet.

## Limitations

- Text-only prompts. Image inputs go through the Transformers encoder path unchanged and are not benchmarked.
- The sampler ported is the released one (entropy bound 0.1, temperature 0.8 to 0.4, at most 48 steps, adaptive
  stopping). The other Diffusers schedulers are not ported.
- Tokens differ from a GPU run of Transformers because the random streams differ; equality is tested on CPU with
  shared noise.
- The autoregressive baseline is plain decoding without multi-token speculative decoding, which is what the
  DiffusionGemma report compares against on GPU.
