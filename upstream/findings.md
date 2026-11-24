### Upstream findings

Drafts only, nothing here has been filed. Versions: torch 2.15.0, torch_tpu 0.2.0 (libtpu 0.0.52), transformers 5.22.2,
TPU v6e. Each entry lists the evidence in this repository. Finding 1 has its own report,
`transformers-static-sliding-cache.md`; the others are below.

| # | Target | Finding | Evidence |
|---|---|---|---|
| 1 | transformers | Static cache decodes from a corrupted sliding window past `sliding_window` tokens | `transformers-static-sliding-cache.md`, with a CPU reproduction |
| 2 | transformers | `generate(cache_implementation="static")` cannot run on TPU | `results/session1/proxy_transformers_static.log` |
| 3 | transformers | Default experts implementation is far from optimal on TPU | `results/session1/moe_formulations.jsonl`, `results/session1b/results.jsonl` |
| 4 | transformers | Tensor- and expert-parallel plans fail for DiffusionGemma | Failing configurations and errors below; no script in this repository |
| 5 | torch_tpu / transformers | Moving DiffusionGemma to TPU unties encoder and decoder weights | `results/session1b/results.jsonl`, `results/session1c/results.jsonl` |
| 6 | torch_tpu | Storage-offset guard recompiles a step for every slice of one tensor | Guard output below |

### 2. Compiled DiffusionGemma generation fails on TPU

`DiffusionGemmaGenerationMixin._compile_functions` hard-codes `torch.compile(..., mode="reduce-overhead", fullgraph=True)`
for the encoder, the decoder forward and the sampler's `accept_canvas`, and `_denoising_step` calls
`torch.compiler.cudagraph_mark_step_begin()`. The TPU backend rejects the mode:

```
torch._dynamo.exc.BackendCompilerFailed: backend='_default_backend_selector' raised:
TypeError: Unexpected keyword arguments: {'mode': 'reduce-overhead'}
```

`PreTrainedModel._default_compile_config` already returns `CompileConfig(backend="tpu", dynamic=False, mode="default")`
when the model sits on a TPU, and autoregressive `generate` uses it; the diffusion mixin does not. Only the uncompiled
dynamic-cache path runs (`results/session1/proxy_transformers_dynamic.log`).

### 3. Experts implementation on TPU

One layer of the released shape (128 experts, hidden size 2816, expert intermediate size 704, eight experts per token),
compiled, bf16, milliseconds. The dense column is `dense_preweighted` in `dgemma_tpu/moe.py`, the form the selector
uses; the plain dense variant is within 0.05 ms of it up to 256 tokens and slower beyond.

| Tokens in the pass | `grouped_mm` (default) | `batched_mm` | dense, all experts |
|---|---|---|---|
| 1 | 3.10 | 0.13 | 1.31 |
| 8 | 3.08 | 1.92 | 1.32 |
| 256 | 3.44 | 64.98 | 1.38 |
| 1024 | 3.74 | compile-time HBM OOM (61.9 GB of temporaries against 31.2 GB) | 2.37 |
| 2048 | 5.27 | compile-time HBM OOM (122.5 GB of temporaries) | 4.96 |

`grouped_mm` costs about 3 ms regardless of the token count. An XProf trace of the denoising step
(`results/session1/xprof_denoise_proxy6_grouped_mm.trace.json.gz`) shows a copy of the full `bf16[128,1408,2816]`
expert tensor into another layout and offloaded gathers on every step. Selecting by token count (`dgemma_tpu/moe.py`)
takes a 6-layer autoregressive decode step from 20.8 ms to 2.7 ms and a 256-token denoising step from 26.6 ms to
15.9 ms (`results/session1b`, batch 1).

### 4. Parallel plans for DiffusionGemma

Two CPU ranks with the `gloo` backend, tiny random model, `DiffusionGemmaForBlockDiffusion.from_pretrained(...,
distributed_config=...)` followed by `generate`. These runs were done by hand and are not scripted in this repository.

| `DistributedConfig` | Result |
|---|---|
| `tp_plan="auto"` | `RuntimeError: aten.embedding.default got mixed torch.Tensor and DTensor` in the encoder embedding |
| `tp_plan="auto"`, class `_tp_plan = {}` | same error |
| `ep_size=2, tp_size=2, tp_plan={}` | `RuntimeError: matrix batch sizes have to match` in `grouped_mm` |

After loading with `tp_plan="auto"` the text layers keep full-size weights while the vision tower's projections become
`DTensor`s, which suggests the `layers.*` patterns of the text plan resolve against the vision tower. This repository
shards only the experts along their intermediate dimension and all-reduces once per layer (`dgemma_tpu/dist.py`).

### 5. Weight ties are lost on `.to("tpu")`

The encoder and decoder of DiffusionGemma share every transformer parameter object. A 6-layer model with the released
width holds 11.6 GiB of distinct weights in bf16. After `model.to("tpu")` it occupies 24.1 GiB of HBM with the
token-count experts and 24.6 GiB with `grouped_mm` (`results/session1b`, diffusion at batch 1). Moving each distinct
parameter once and reassigning it to every module that references it (`dgemma_tpu.runtime.move_preserving_ties`)
brings that to 12.2 GiB at the same speed, 366.6 against 367.0 tokens per second (`results/session1c`,
`diffusion_proxy6_tied`). The released 26B model would need about 96 GiB instead of 48 GiB. torch_tpu issue 4161
describes the same symptom; it was closed as completed on 2026-10-16 (pull request 5412), and the duplication was
still observed here with torch_tpu 0.2.0 on 2026-11-10.

### 6. Storage-offset guard

Passing `temperatures[i]` for `temperatures = torch.tensor(schedule, device="tpu").unbind(0)` into a function compiled
with `backend="tpu"` adds a guard on `storage_offset()` and recompiles for every index:

```
Recompiling function _denoise_step_impl
    triggered by the following guard failure(s):
    - 1/1: temperature.storage_offset() == 1
    - 1/0: temperature.storage_offset() == 0
```

The same code compiles once with `aot_eager` on CPU. Creating one tensor per value avoids it, which is what
`StaticBlockDiffusionGenerator` does for its temperature schedule (`dgemma_tpu/generate.py`). torch_tpu issue 5000
reports the same guard and is open.
