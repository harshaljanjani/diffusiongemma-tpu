import math
import time
from dataclasses import dataclass

import torch

from .cache import BlockDiffusionStaticCache, to_attention_bias
from .runtime import synchronize
from .sampler import (
    DeviceNoise,
    entropy_bound_acceptance,
    finalize_canvas,
    gumbel_max_sample,
    token_distribution,
)


@dataclass
class BlockDiffusionOutput:
    """
    Output of [`StaticBlockDiffusionGenerator.generate`].

    Args:
        sequences (`torch.Tensor` of shape `(batch_size, prompt_length + num_canvases * canvas_length)`):
            Prompt followed by the generated tokens, padded after the end-of-sequence token.
        decoder_forward_passes (`torch.Tensor` of shape `(batch_size,)`):
            Denoising steps each sequence needed before it stopped changing.
        num_canvases (`int`):
            Number of canvases generated.
        num_denoising_steps (`int`):
            Number of decoder passes executed for the batch.
        num_stop_checks (`int`):
            Number of host synchronizations spent on stopping checks.
        prompt_length (`int`):
            Length of the prompt passed by the caller.
        pad_token_id (`int`):
            Padding token id used in `sequences`.
        prefill_seconds (`float`, *optional*):
            Wall-clock time of the prompt prefill, set when profiling.
        canvas_seconds (`list[float]`, *optional*):
            Wall-clock time of each canvas, set when profiling.
    """

    sequences: torch.Tensor
    decoder_forward_passes: torch.Tensor
    num_canvases: int
    num_denoising_steps: int
    num_stop_checks: int
    prompt_length: int
    pad_token_id: int
    prefill_seconds: float | None = None
    canvas_seconds: list[float] | None = None

    @property
    def tokens_per_forward(self) -> torch.Tensor:
        """
        Returns:
            `torch.Tensor`: Generated non-padding tokens per decoder pass, of shape `(batch_size,)`.
        """
        new_tokens = self.sequences[:, self.prompt_length :]
        num_tokens = (new_tokens != self.pad_token_id).sum(dim=-1)
        return num_tokens / self.decoder_forward_passes.clamp(min=1)


@dataclass
class _GenerationState:
    cache: BlockDiffusionStaticCache
    sequences: torch.Tensor
    initial_soft_embeddings: torch.Tensor
    initial_finished: torch.Tensor
    initial_history: torch.Tensor


class StaticBlockDiffusionGenerator:
    """
    Static-shape block-diffusion decoding for DiffusionGemma with the released entropy-bound sampler.

    Every tensor keeps a fixed shape for a given batch size and prompt bucket, and the denoising loop only
    synchronizes with the host every `stop_check_interval` steps, so each stage compiles once.

    Args:
        model (`DiffusionGemmaForBlockDiffusion`):
            Model to decode with.
        max_new_tokens (`int`, *optional*, defaults to 256):
            Maximum number of generated tokens, rounded up to a whole number of canvases.
        prompt_bucket (`int`, *optional*, defaults to 128):
            Prompts are left-padded to a multiple of this length.
        stop_check_interval (`int`, *optional*, defaults to 4):
            Number of denoising steps between adaptive-stopping checks. `0` disables the checks.
        compile_backend (`str` or `Callable`, *optional*):
            Backend passed to `torch.compile`. Runs eagerly when unset.
        compile_options (`dict`, *optional*):
            Extra keyword arguments for `torch.compile`.
        noise (`DeviceNoise`, *optional*):
            Source of sampling noise.
        generation_kwargs:
            Overrides for the fields of `DiffusionGemmaGenerationConfig`.
    """

    def __init__(
        self,
        model,
        max_new_tokens: int = 256,
        prompt_bucket: int = 128,
        stop_check_interval: int = 4,
        compile_backend=None,
        compile_options: dict | None = None,
        noise: DeviceNoise | None = None,
        **generation_kwargs,
    ):
        self.model = model
        self.prompt_bucket = prompt_bucket
        self.stop_check_interval = stop_check_interval
        self.noise = noise if noise is not None else DeviceNoise()

        text_config = model.config.text_config
        decoder = model.model.decoder
        self.text_config = text_config
        self.canvas_length = model.config.canvas_length
        self.vocab_size = text_config.vocab_size
        self.num_canvases = math.ceil(max_new_tokens / self.canvas_length)
        self.embedding_dtype = decoder.embed_tokens.weight.dtype
        self.attn_implementation = decoder.layers[0].self_attn.config._attn_implementation

        generation_config, _ = model._prepare_generation_config(None, **generation_kwargs)
        self.max_denoising_steps = generation_config.max_denoising_steps
        self.entropy_bound = generation_config.sampler_config.entropy_bound
        self.stability_threshold = generation_config.stability_threshold
        self.confidence_threshold = generation_config.confidence_threshold
        self.pad_token_id = generation_config.pad_token_id if generation_config.pad_token_id is not None else 0
        eos_token_id = generation_config.eos_token_id
        if eos_token_id is None:
            eos_token_id = []
        self.eos_token_ids = list(eos_token_id) if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
        self.temperature_schedule = [
            generation_config.t_min
            + (generation_config.t_max - generation_config.t_min) * (remaining_steps / self.max_denoising_steps)
            for remaining_steps in range(self.max_denoising_steps, 0, -1)
        ]

        self._states = {}
        self._prefill = self._prefill_impl
        self._denoise_step = self._denoise_step_impl
        self._finalize = self._finalize_impl
        self._extend_cache = self._extend_cache_impl
        if compile_backend is not None:
            options = {"fullgraph": True, "dynamic": False, **(compile_options or {})}
            self._prefill = torch.compile(self._prefill_impl, backend=compile_backend, **options)
            self._denoise_step = torch.compile(self._denoise_step_impl, backend=compile_backend, **options)
            self._finalize = torch.compile(self._finalize_impl, backend=compile_backend, **options)
            self._extend_cache = torch.compile(self._extend_cache_impl, backend=compile_backend, **options)

    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, profile: bool = False
    ) -> BlockDiffusionOutput:
        """
        Generates up to `max_new_tokens` tokens for a batch of text prompts.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, prompt_length)`):
                Prompt token ids.
            attention_mask (`torch.Tensor` of shape `(batch_size, prompt_length)`, *optional*):
                Mask that is 0 for prompt padding.
            profile (`bool`, *optional*, defaults to `False`):
                Whether to time the prefill and each canvas, which adds one host synchronization per stage.

        Returns:
            [`BlockDiffusionOutput`]: Generated sequences and decoding statistics.
        """
        device = input_ids.device
        batch_size, prompt_length = input_ids.shape
        canvas_length = self.canvas_length
        bucketed_length = math.ceil(prompt_length / self.prompt_bucket) * self.prompt_bucket
        left_padding = bucketed_length - prompt_length
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        input_ids = torch.nn.functional.pad(input_ids, (left_padding, 0), value=self.pad_token_id)
        token_valid = torch.nn.functional.pad(attention_mask.bool(), (left_padding, 0), value=False)

        state = self._get_state(batch_size, bucketed_length + self.num_canvases * canvas_length, device)
        eos_token_ids = torch.tensor(self.eos_token_ids, dtype=torch.long, device=device)
        temperatures = torch.tensor(self.temperature_schedule, dtype=torch.float32, device=device).unbind(0)
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
        forward_passes = torch.zeros(batch_size, dtype=torch.int32, device=device)

        stage_start = _stage_clock(device, profile)
        position_ids = self._prefill(state, input_ids, token_valid)
        prefill_seconds = _stage_elapsed(device, profile, stage_start)
        canvas_seconds = []

        num_canvases = 0
        num_denoising_steps = 0
        num_stop_checks = 0
        for canvas_index in range(self.num_canvases):
            stage_start = _stage_clock(device, profile)
            canvas = self.noise.tokens((batch_size, canvas_length), self.vocab_size, device)
            # A distinct tensor avoids an input-aliasing guard that would compile the first step separately.
            argmax_canvas = canvas.clone()
            soft_embeddings = state.initial_soft_embeddings
            finished = state.initial_finished
            history = state.initial_history
            for step_index, temperature in enumerate(temperatures):
                canvas, argmax_canvas, soft_embeddings, finished, history, forward_passes = self._denoise_step(
                    state,
                    canvas,
                    argmax_canvas,
                    soft_embeddings,
                    finished,
                    history,
                    forward_passes,
                    finished_sequences,
                    temperature,
                    position_ids,
                )
                num_denoising_steps += 1
                is_check_step = self.stop_check_interval > 0 and (step_index + 1) % self.stop_check_interval == 0
                if is_check_step and step_index + 1 < self.max_denoising_steps:
                    num_stop_checks += 1
                    if bool(finished.all()):
                        break

            committed_canvas, finished_sequences = self._finalize(
                state, argmax_canvas, finished_sequences, eos_token_ids
            )
            num_canvases += 1
            is_last_canvas = canvas_index + 1 == self.num_canvases or bool(finished_sequences.all())
            if not is_last_canvas:
                self._extend_cache(state, committed_canvas, position_ids)
                position_ids = position_ids + canvas_length
            canvas_seconds.append(_stage_elapsed(device, profile, stage_start))
            if is_last_canvas:
                break

        generated_end = bucketed_length + num_canvases * canvas_length
        return BlockDiffusionOutput(
            sequences=state.sequences[:, left_padding:generated_end].clone(),
            decoder_forward_passes=forward_passes,
            num_canvases=num_canvases,
            num_denoising_steps=num_denoising_steps,
            num_stop_checks=num_stop_checks,
            prompt_length=prompt_length,
            pad_token_id=self.pad_token_id,
            prefill_seconds=prefill_seconds,
            canvas_seconds=canvas_seconds if profile else None,
        )

    def _get_state(self, batch_size: int, max_length: int, device: torch.device) -> _GenerationState:
        key = (batch_size, max_length)
        if key in self._states:
            state = self._states[key]
            state.cache.reset()
            state.sequences.fill_(self.pad_token_id)
            return state

        cache = BlockDiffusionStaticCache(self.text_config, batch_size, max_length, device)
        encoder_layers = self.model.model.encoder.language_model.layers
        for cache_layer, encoder_layer in zip(cache.layers, encoder_layers):
            attention = encoder_layer.self_attn
            num_heads = attention.k_proj.weight.shape[0] // attention.head_dim
            cache_layer.allocate(batch_size, num_heads, attention.head_dim, self.embedding_dtype, device)

        history_length = max(self.stability_threshold, 1)
        state = _GenerationState(
            cache=cache,
            sequences=torch.full((batch_size, max_length), self.pad_token_id, dtype=torch.long, device=device),
            initial_soft_embeddings=torch.zeros(
                batch_size, self.canvas_length, self.text_config.hidden_size, dtype=self.embedding_dtype, device=device
            ),
            initial_finished=torch.zeros(batch_size, dtype=torch.bool, device=device),
            initial_history=torch.full(
                (history_length, batch_size, self.canvas_length), -1, dtype=torch.long, device=device
            ),
        )
        self._states[key] = state
        return state

    def _attention_bias(self, masks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return to_attention_bias(masks, self.embedding_dtype, self.attn_implementation)

    def _prefill_impl(self, state: _GenerationState, input_ids: torch.Tensor, token_valid: torch.Tensor):
        cache = state.cache
        prompt_slots = torch.arange(input_ids.shape[1], device=input_ids.device)
        state.sequences.index_copy_(1, prompt_slots, input_ids)
        num_valid_before = token_valid.cumsum(dim=-1)
        position_ids = (num_valid_before - 1).clamp(min=0).to(torch.int32)
        self.model.model.encoder(
            input_ids=input_ids,
            attention_mask=self._attention_bias(cache.encoder_attention_masks(token_valid)),
            past_key_values=cache,
            position_ids=position_ids,
        )
        cache.advance(token_valid)
        canvas_offsets = torch.arange(self.canvas_length, device=input_ids.device)
        return (num_valid_before[:, -1:] + canvas_offsets[None]).to(torch.int32)

    def _decode_canvas(
        self, cache: BlockDiffusionStaticCache, canvas: torch.Tensor, soft_embeddings: torch.Tensor, position_ids
    ) -> torch.Tensor:
        decoder = self.model.model.decoder
        layer_types = self.text_config.layer_types
        masks = self._attention_bias(cache.decoder_attention_masks(self.canvas_length))
        hidden_states = decoder.self_conditioning(decoder.embed_tokens(canvas), soft_embeddings)
        position_embeddings = {
            layer_type: decoder.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in decoder.unique_layer_types
        }
        for layer_index, decoder_layer in enumerate(decoder.layers[: self.text_config.num_hidden_layers]):
            layer_type = layer_types[layer_index]
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[layer_type],
                attention_mask=masks[layer_type],
                position_ids=position_ids,
                past_key_values=cache,
            )
        logits = self.model.lm_head(decoder.norm(hidden_states)).to(torch.float32)
        softcapping = self.model.final_logit_softcapping
        if softcapping is not None:
            logits = torch.tanh(logits / softcapping) * softcapping
        return logits

    def _denoise_step_impl(
        self,
        state: _GenerationState,
        canvas: torch.Tensor,
        argmax_canvas: torch.Tensor,
        soft_embeddings: torch.Tensor,
        finished: torch.Tensor,
        history: torch.Tensor,
        forward_passes: torch.Tensor,
        finished_sequences: torch.Tensor,
        temperature: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        logits = self._decode_canvas(state.cache, canvas, soft_embeddings, position_ids) / temperature
        log_probs, probs, token_entropy = token_distribution(logits)
        sampled_canvas = gumbel_max_sample(log_probs, self.noise.uniform(log_probs.shape, log_probs.device))
        new_argmax_canvas = torch.argmax(logits, dim=-1)
        accepted = entropy_bound_acceptance(token_entropy, self.entropy_bound)
        random_canvas = self.noise.tokens(canvas.shape, self.vocab_size, canvas.device)
        new_canvas = torch.where(accepted, sampled_canvas, random_canvas)

        # Carrying the soft embeddings instead of the logits keeps a (batch, canvas, vocab) tensor out of the
        # step boundary; the decoder would recompute exactly this product from the logits on the next step.
        embedding = self.model.model.decoder.embed_tokens
        new_soft_embeddings = torch.matmul(probs.to(embedding.weight.dtype), embedding.weight)
        new_soft_embeddings = new_soft_embeddings * embedding.embed_scale.to(new_soft_embeddings.dtype)

        forward_passes = forward_passes + (~(finished | finished_sequences)).to(forward_passes.dtype)
        frozen = finished[:, None]
        new_canvas = torch.where(frozen, canvas, new_canvas)
        new_argmax_canvas = torch.where(frozen, argmax_canvas, new_argmax_canvas)
        new_soft_embeddings = torch.where(frozen[:, :, None], soft_embeddings, new_soft_embeddings)

        confident = token_entropy.mean(dim=-1) < self.confidence_threshold
        if self.stability_threshold > 0:
            stable = (history == new_argmax_canvas[None]).all(dim=-1).all(dim=0)
            history = torch.cat([history[1:], new_argmax_canvas[None]], dim=0)
            new_finished = finished | (stable & confident)
        else:
            new_finished = finished | confident
        return new_canvas, new_argmax_canvas, new_soft_embeddings, new_finished, history, forward_passes

    def _finalize_impl(
        self,
        state: _GenerationState,
        argmax_canvas: torch.Tensor,
        finished_sequences: torch.Tensor,
        eos_token_ids: torch.Tensor,
    ):
        committed_canvas, finished_sequences = finalize_canvas(
            argmax_canvas, finished_sequences, eos_token_ids, self.pad_token_id
        )
        canvas_slots = torch.arange(self.canvas_length, device=argmax_canvas.device) + state.cache.length
        state.sequences.index_copy_(1, canvas_slots, committed_canvas)
        return committed_canvas, finished_sequences

    def _extend_cache_impl(self, state: _GenerationState, committed_canvas: torch.Tensor, position_ids: torch.Tensor):
        cache = state.cache
        token_valid = torch.ones_like(committed_canvas, dtype=torch.bool)
        self.model.model.encoder(
            input_ids=committed_canvas,
            attention_mask=self._attention_bias(cache.encoder_attention_masks(token_valid)),
            past_key_values=cache,
            position_ids=position_ids,
        )
        cache.advance(token_valid)


def _stage_clock(device: torch.device, profile: bool) -> float | None:
    if not profile:
        return None
    synchronize(device)
    return time.perf_counter()


def _stage_elapsed(device: torch.device, profile: bool, start: float | None) -> float | None:
    if not profile:
        return None
    synchronize(device)
    return time.perf_counter() - start
