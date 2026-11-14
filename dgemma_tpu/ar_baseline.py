import math
import time
from dataclasses import dataclass

import torch

from .cache import BlockDiffusionStaticCache, to_attention_bias
from .runtime import synchronize
from .sampler import DeviceNoise, gumbel_max_sample


@dataclass
class AutoregressiveOutput:
    """
    Output of [`StaticAutoregressiveGenerator.generate`].

    Args:
        sequences (`torch.Tensor` of shape `(batch_size, prompt_length + num_steps)`):
            Prompt followed by the generated tokens, padded after the end-of-sequence token.
        num_decode_steps (`int`):
            Number of single-token decoder passes executed.
        num_stop_checks (`int`):
            Number of host synchronizations spent on stopping checks.
        prompt_length (`int`):
            Length of the prompt passed by the caller.
        prefill_seconds (`float`, *optional*):
            Wall-clock time of the prompt prefill including the first token, set when profiling.
    """

    sequences: torch.Tensor
    num_decode_steps: int
    num_stop_checks: int
    prompt_length: int
    prefill_seconds: float | None = None


@dataclass
class _DecodeState:
    cache: BlockDiffusionStaticCache
    sequences: torch.Tensor


class StaticAutoregressiveGenerator:
    """
    Static-shape autoregressive decoding for Gemma 4 that shares the cache and stopping policy of the diffusion sampler.

    Args:
        model (`Gemma4ForConditionalGeneration`):
            Model to decode with.
        max_new_tokens (`int`, *optional*, defaults to 256):
            Maximum number of generated tokens.
        prompt_bucket (`int`, *optional*, defaults to 128):
            Prompts are left-padded to a multiple of this length.
        stop_check_interval (`int`, *optional*, defaults to 16):
            Number of decode steps between end-of-sequence checks. `0` disables the checks.
        do_sample (`bool`, *optional*, defaults to `False`):
            Whether to sample instead of taking the most likely token.
        temperature (`float`, *optional*, defaults to 1.0):
            Sampling temperature.
        top_k (`int`, *optional*, defaults to 64):
            Number of highest-probability tokens kept when sampling.
        compile_backend (`str` or `Callable`, *optional*):
            Backend passed to `torch.compile`. Runs eagerly when unset.
        compile_options (`dict`, *optional*):
            Extra keyword arguments for `torch.compile`.
        function_decorator (`Callable`, *optional*):
            Decorator applied to each stage before compilation, such as `torch.tpu.distributed.spmd_safe`.
        noise (`DeviceNoise`, *optional*):
            Source of sampling noise.
    """

    def __init__(
        self,
        model,
        max_new_tokens: int = 256,
        prompt_bucket: int = 128,
        stop_check_interval: int = 16,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 64,
        compile_backend=None,
        compile_options: dict | None = None,
        function_decorator=None,
        noise: DeviceNoise | None = None,
    ):
        self.model = model
        self.text_model = model.model.language_model
        self.text_config = model.config.text_config
        self.max_new_tokens = max_new_tokens
        self.prompt_bucket = prompt_bucket
        self.stop_check_interval = stop_check_interval
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.noise = noise if noise is not None else DeviceNoise()

        first_attention = self.text_model.layers[0].self_attn
        self.embedding_dtype = self.text_model.embed_tokens.weight.dtype
        self.attn_implementation = first_attention.config._attn_implementation
        self.softcapping = getattr(self.text_config, "final_logit_softcapping", None)
        generation_config = model.generation_config
        self.pad_token_id = generation_config.pad_token_id if generation_config.pad_token_id is not None else 0
        eos_token_id = generation_config.eos_token_id
        if eos_token_id is None:
            eos_token_id = []
        self.eos_token_ids = list(eos_token_id) if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]

        self._states = {}
        self._prefill = self._prefill_impl
        self._decode_step = self._decode_step_impl
        if compile_backend is not None:
            options = {"fullgraph": True, "dynamic": False, **(compile_options or {})}
            decorate = function_decorator if function_decorator is not None else lambda function: function
            self._prefill = torch.compile(decorate(self._prefill_impl), backend=compile_backend, **options)
            self._decode_step = torch.compile(decorate(self._decode_step_impl), backend=compile_backend, **options)

    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, profile: bool = False
    ) -> AutoregressiveOutput:
        """
        Generates up to `max_new_tokens` tokens for a batch of text prompts, one token per decoder pass.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, prompt_length)`):
                Prompt token ids.
            attention_mask (`torch.Tensor` of shape `(batch_size, prompt_length)`, *optional*):
                Mask that is 0 for prompt padding.
            profile (`bool`, *optional*, defaults to `False`):
                Whether to time the prefill, which adds one host synchronization.

        Returns:
            [`AutoregressiveOutput`]: Generated sequences and decoding statistics.
        """
        device = input_ids.device
        batch_size, prompt_length = input_ids.shape
        bucketed_length = math.ceil(prompt_length / self.prompt_bucket) * self.prompt_bucket
        left_padding = bucketed_length - prompt_length
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        input_ids = torch.nn.functional.pad(input_ids, (left_padding, 0), value=self.pad_token_id)
        token_valid = torch.nn.functional.pad(attention_mask.bool(), (left_padding, 0), value=False)

        state = self._get_state(batch_size, bucketed_length + self.max_new_tokens, device)
        eos_token_ids = torch.tensor(self.eos_token_ids, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if profile:
            synchronize(device)
            start = time.perf_counter()
        token, position_ids, finished = self._prefill(state, input_ids, token_valid, finished, eos_token_ids)
        prefill_seconds = None
        if profile:
            synchronize(device)
            prefill_seconds = time.perf_counter() - start

        num_decode_steps = 0
        num_stop_checks = 0
        for step_index in range(1, self.max_new_tokens):
            token, position_ids, finished = self._decode_step(state, token, position_ids, finished, eos_token_ids)
            num_decode_steps += 1
            is_check_step = self.stop_check_interval > 0 and step_index % self.stop_check_interval == 0
            if is_check_step and step_index + 1 < self.max_new_tokens:
                num_stop_checks += 1
                if bool(finished.all()):
                    break

        generated_end = bucketed_length + num_decode_steps + 1
        return AutoregressiveOutput(
            sequences=state.sequences[:, left_padding:generated_end].clone(),
            num_decode_steps=num_decode_steps + 1,
            num_stop_checks=num_stop_checks,
            prompt_length=prompt_length,
            prefill_seconds=prefill_seconds,
        )

    def _get_state(self, batch_size: int, max_length: int, device: torch.device) -> _DecodeState:
        key = (batch_size, max_length)
        if key in self._states:
            state = self._states[key]
            state.cache.reset()
            state.sequences.fill_(self.pad_token_id)
            return state

        cache = BlockDiffusionStaticCache(self.text_config, batch_size, max_length, device)
        for cache_layer, decoder_layer in zip(cache.layers, self.text_model.layers):
            attention = decoder_layer.self_attn
            num_heads = attention.k_proj.weight.shape[0] // attention.head_dim
            cache_layer.allocate(batch_size, num_heads, attention.head_dim, self.embedding_dtype, device)
        sequences = torch.full((batch_size, max_length), self.pad_token_id, dtype=torch.long, device=device)
        state = _DecodeState(cache=cache, sequences=sequences)
        self._states[key] = state
        return state

    def _next_token_logits(self, cache, input_ids, token_valid, position_ids) -> torch.Tensor:
        masks = to_attention_bias(
            cache.encoder_attention_masks(token_valid), self.embedding_dtype, self.attn_implementation
        )
        outputs = self.text_model(
            input_ids=input_ids, attention_mask=masks, position_ids=position_ids, past_key_values=cache
        )
        cache.advance(token_valid)
        logits = self.model.lm_head(outputs.last_hidden_state[:, -1]).to(torch.float32)
        if self.softcapping is not None:
            logits = torch.tanh(logits / self.softcapping) * self.softcapping
        return logits

    def _select_token(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.do_sample:
            return torch.argmax(logits, dim=-1)
        top_logits, top_indices = torch.topk(logits / self.temperature, self.top_k, dim=-1)
        log_probs = torch.log_softmax(top_logits, dim=-1)
        choice = gumbel_max_sample(log_probs, self.noise.uniform(log_probs.shape, log_probs.device))
        return torch.gather(top_indices, -1, choice[:, None]).squeeze(-1)

    def _emit_token(self, state: _DecodeState, logits, finished, eos_token_ids):
        token = self._select_token(logits)
        token = torch.where(finished, torch.full_like(token, self.pad_token_id), token)
        state.sequences.index_copy_(1, state.cache.length[None], token[:, None])
        return token, finished | torch.isin(token, eos_token_ids)

    def _prefill_impl(self, state: _DecodeState, input_ids, token_valid, finished, eos_token_ids):
        prompt_slots = torch.arange(input_ids.shape[1], device=input_ids.device)
        state.sequences.index_copy_(1, prompt_slots, input_ids)
        num_valid_before = token_valid.cumsum(dim=-1)
        position_ids = (num_valid_before - 1).clamp(min=0).to(torch.int32)
        logits = self._next_token_logits(state.cache, input_ids, token_valid, position_ids)
        token, finished = self._emit_token(state, logits, finished, eos_token_ids)
        return token, num_valid_before[:, -1:].to(torch.int32), finished

    def _decode_step_impl(self, state: _DecodeState, token, position_ids, finished, eos_token_ids):
        token_valid = torch.ones_like(token, dtype=torch.bool)[:, None]
        logits = self._next_token_logits(state.cache, token[:, None], token_valid, position_ids)
        token, finished = self._emit_token(state, logits, finished, eos_token_ids)
        return token, position_ids + 1, finished
