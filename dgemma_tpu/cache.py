import torch


class FullAttentionLayer:
    """
    Fixed-size key/value store for a full-attention layer.

    Args:
        max_length (`int`):
            Number of token slots to preallocate.
    """

    def __init__(self, max_length: int):
        self.max_length = max_length
        self.keys = None
        self.values = None

    def allocate(self, batch_size: int, num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device):
        """
        Preallocates the store so that no compiled graph has to create it.

        Args:
            batch_size (`int`):
                Number of sequences.
            num_heads (`int`):
                Number of key/value heads.
            head_dim (`int`):
                Dimension of each head.
            dtype (`torch.dtype`):
                Dtype of the states.
            device (`torch.device`):
                Device of the states.
        """
        shape = (batch_size, num_heads, self.max_length, head_dim)
        self.keys = torch.zeros(shape, dtype=dtype, device=device)
        self.values = torch.zeros(shape, dtype=dtype, device=device)

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Writes new states at slot `length` and returns the whole store.

        Args:
            key_states (`torch.Tensor` of shape `(batch_size, num_heads, num_tokens, head_dim)`):
                Keys of the new tokens.
            value_states (`torch.Tensor` of shape `(batch_size, num_heads, num_tokens, head_dim)`):
                Values of the new tokens.
            length (`torch.Tensor`):
                0-D number of slots already written.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: Keys and values of shape `(batch_size, num_heads, max_length, head_dim)`.
        """
        if self.keys is None:
            self.keys = _allocate(key_states, self.max_length)
            self.values = _allocate(value_states, self.max_length)
        slots = torch.arange(key_states.shape[2], device=key_states.device) + length
        self.keys.index_copy_(2, slots, key_states)
        self.values.index_copy_(2, slots, value_states)
        return self.keys, self.values


class SlidingAttentionLayer:
    """
    Fixed-size key/value store holding the most recent `window_length` tokens, right-aligned.

    Args:
        window_length (`int`):
            Number of trailing token slots to keep.
    """

    def __init__(self, window_length: int):
        self.window_length = window_length
        self.keys = None
        self.values = None

    def allocate(self, batch_size: int, num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device):
        """
        Preallocates the store so that no compiled graph has to create it.

        Args:
            batch_size (`int`):
                Number of sequences.
            num_heads (`int`):
                Number of key/value heads.
            head_dim (`int`):
                Dimension of each head.
            dtype (`torch.dtype`):
                Dtype of the states.
            device (`torch.device`):
                Device of the states.
        """
        shape = (batch_size, num_heads, self.window_length, head_dim)
        self.keys = torch.zeros(shape, dtype=dtype, device=device)
        self.values = torch.zeros(shape, dtype=dtype, device=device)

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Appends new states, keeps the trailing window and returns window plus new states.

        Args:
            key_states (`torch.Tensor` of shape `(batch_size, num_heads, num_tokens, head_dim)`):
                Keys of the new tokens.
            value_states (`torch.Tensor` of shape `(batch_size, num_heads, num_tokens, head_dim)`):
                Values of the new tokens.
            length (`torch.Tensor`):
                Unused, kept for a uniform layer interface.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: Keys and values of shape
            `(batch_size, num_heads, window_length + num_tokens, head_dim)`.
        """
        if self.keys is None:
            self.keys = _allocate(key_states, self.window_length)
            self.values = _allocate(value_states, self.window_length)
        keys = torch.cat([self.keys, key_states], dim=2)
        values = torch.cat([self.values, value_states], dim=2)
        self.keys.copy_(keys[:, :, -self.window_length :])
        self.values.copy_(values[:, :, -self.window_length :])
        return keys, values


class BlockDiffusionStaticCache:
    """
    Static-shape KV cache for DiffusionGemma whose length and validity live in device tensors.

    Every update has the same tensor shapes regardless of how many tokens are cached, so one compiled graph serves
    all canvases. Sliding layers keep `sliding_window - 1` trailing tokens, matching the dynamic cache of Transformers.

    Args:
        text_config (`DiffusionGemmaTextConfig`):
            Text configuration providing `layer_types` and `sliding_window`.
        batch_size (`int`):
            Number of sequences.
        max_length (`int`):
            Maximum number of cached tokens, prompt padding included.
        device (`torch.device`):
            Device holding the cache state.
    """

    is_compileable = False

    def __init__(self, text_config, batch_size: int, max_length: int, device: torch.device):
        self.batch_size = batch_size
        self.max_length = max_length
        self.sliding_window = text_config.sliding_window
        self.window_length = min(self.sliding_window - 1, max_length)
        self.is_sliding = [layer_type == "sliding_attention" for layer_type in text_config.layer_types]
        self.layers = [
            SlidingAttentionLayer(self.window_length) if is_sliding else FullAttentionLayer(max_length)
            for is_sliding in self.is_sliding
        ]
        self.length = torch.zeros((), dtype=torch.long, device=device)
        self.slot_valid = torch.zeros(batch_size, max_length, dtype=torch.bool, device=device)
        self.window_valid = torch.zeros(batch_size, self.window_length, dtype=torch.bool, device=device)

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Stores the new states of one layer and returns the keys and values to attend over.

        Args:
            key_states (`torch.Tensor`):
                Keys of the new tokens.
            value_states (`torch.Tensor`):
                Values of the new tokens.
            layer_idx (`int`):
                Index of the layer being updated.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: Keys and values for the attention of this layer.
        """
        return self.layers[layer_idx].update(key_states, value_states, self.length)

    def advance(self, token_valid: torch.Tensor) -> None:
        """
        Marks the tokens written by the last encoder pass as cached.

        Args:
            token_valid (`torch.Tensor` of shape `(batch_size, num_tokens)`):
                Boolean mask that is `False` for padding tokens.
        """
        num_tokens = token_valid.shape[1]
        slots = torch.arange(num_tokens, device=token_valid.device) + self.length
        self.slot_valid.index_copy_(1, slots, token_valid)
        self.window_valid.copy_(torch.cat([self.window_valid, token_valid], dim=1)[:, -self.window_length :])
        self.length.add_(num_tokens)

    def reset(self) -> None:
        """Empties the cache in place while keeping its allocated buffers."""
        self.length.zero_()
        self.slot_valid.zero_()
        self.window_valid.zero_()

    def get_seq_length(self, layer_idx: int = 0) -> torch.Tensor:
        """
        Returns the number of cached token slots.

        Args:
            layer_idx (`int`, *optional*, defaults to 0):
                Unused, kept for compatibility with the Transformers cache interface.

        Returns:
            `torch.Tensor`: 0-D length tensor.
        """
        return self.length

    def encoder_attention_masks(self, token_valid: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Builds the causal masks for an encoder pass over new tokens that start at slot `length`.

        Args:
            token_valid (`torch.Tensor` of shape `(batch_size, num_tokens)`):
                Boolean mask that is `False` for padding tokens.

        Returns:
            `dict[str, torch.Tensor]`: Boolean masks of shape `(batch_size, 1, num_tokens, num_keys)` per layer type,
            where `True` means attend.
        """
        num_tokens = token_valid.shape[1]
        device = token_valid.device
        query_index = torch.arange(num_tokens, device=device)
        query_slot = query_index + self.length

        key_slot = torch.arange(self.max_length, device=device)
        valid_after = self.slot_valid.index_copy(1, query_slot, token_valid)
        is_self = key_slot[None, :] == query_slot[:, None]
        causal = key_slot[None, :] <= query_slot[:, None]
        full_mask = (valid_after[:, None, :] & causal[None]) | is_self[None]

        window_index = torch.arange(self.window_length, device=device)
        window_distance = self.window_length + query_index[:, None] - window_index[None, :]
        window_mask = self.window_valid[:, None, :] & (window_distance < self.sliding_window)[None]
        new_distance = query_index[:, None] - query_index[None, :]
        new_is_self = (new_distance == 0)[None]
        new_in_range = ((new_distance >= 0) & (new_distance < self.sliding_window))[None]
        new_mask = (token_valid[:, None, :] & new_in_range) | new_is_self
        sliding_mask = torch.cat([window_mask, new_mask], dim=-1)

        return {"full_attention": full_mask[:, None], "sliding_attention": sliding_mask[:, None]}

    def decoder_attention_masks(self, canvas_length: int) -> dict[str, torch.Tensor]:
        """
        Builds the bidirectional masks for a decoder pass that reads the cache and attends within the canvas.

        Args:
            canvas_length (`int`):
                Number of canvas tokens.

        Returns:
            `dict[str, torch.Tensor]`: Boolean masks of shape `(batch_size, 1, canvas_length, num_keys)` per layer
            type, where `True` means attend.
        """
        canvas_valid = torch.ones(self.batch_size, canvas_length, dtype=torch.bool, device=self.length.device)
        full_mask = torch.cat([self.slot_valid, canvas_valid], dim=1)
        sliding_mask = torch.cat([self.window_valid, canvas_valid], dim=1)
        return {
            "full_attention": full_mask[:, None, None, :].expand(-1, -1, canvas_length, -1),
            "sliding_attention": sliding_mask[:, None, None, :].expand(-1, -1, canvas_length, -1),
        }


def _allocate(states: torch.Tensor, num_slots: int) -> torch.Tensor:
    batch_size, num_heads, _, head_dim = states.shape
    return torch.zeros(batch_size, num_heads, num_slots, head_dim, dtype=states.dtype, device=states.device)


def to_attention_bias(masks: dict[str, torch.Tensor], dtype: torch.dtype, attn_implementation: str) -> dict:
    """
    Converts boolean masks to the format expected by the attention implementation.

    Args:
        masks (`dict[str, torch.Tensor]`):
            Boolean masks where `True` means attend.
        dtype (`torch.dtype`):
            Floating dtype of the model.
        attn_implementation (`str`):
            Attention implementation name from the model config.

    Returns:
        `dict[str, torch.Tensor]`: Boolean masks for SDPA-style kernels, additive float masks for `"eager"`.
    """
    if attn_implementation != "eager":
        return masks
    minimum = torch.finfo(dtype).min
    return {
        name: torch.where(mask, torch.zeros((), dtype=dtype, device=mask.device), minimum)
        for name, mask in masks.items()
    }
