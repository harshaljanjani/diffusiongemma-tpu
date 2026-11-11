import torch


class DeviceNoise:
    """Draws sampling noise on the target device from the default generator."""

    def uniform(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        """
        Draws uniform samples in `[0, 1)`.

        Args:
            shape (`tuple[int, ...]`):
                Shape of the samples.
            device (`torch.device`):
                Device of the samples.

        Returns:
            `torch.Tensor`: Float32 samples.
        """
        return torch.rand(shape, device=device, dtype=torch.float32)

    def tokens(self, shape: tuple[int, ...], vocab_size: int, device: torch.device) -> torch.Tensor:
        """
        Draws uniformly random token ids.

        Args:
            shape (`tuple[int, ...]`):
                Shape of the samples.
            vocab_size (`int`):
                Exclusive upper bound of the ids.
            device (`torch.device`):
                Device of the samples.

        Returns:
            `torch.Tensor`: Int64 token ids.
        """
        return torch.randint(low=0, high=vocab_size, size=shape, device=device)


def token_distribution(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the per-token distribution statistics with a single normalization pass.

    Args:
        logits (`torch.Tensor` of shape `(batch_size, canvas_length, vocab_size)`):
            Temperature-scaled logits.

    Returns:
        `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: Log-probabilities, probabilities and the entropy of shape
        `(batch_size, canvas_length)`.
    """
    log_probs = torch.log_softmax(logits.to(torch.float32), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return log_probs, probs, entropy


def gumbel_max_sample(log_probs: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
    """
    Samples one token per position with the Gumbel-max trick.

    Args:
        log_probs (`torch.Tensor` of shape `(..., vocab_size)`):
            Log-probabilities.
        uniform (`torch.Tensor` of shape `(..., vocab_size)`):
            Uniform samples in `[0, 1)`.

    Returns:
        `torch.Tensor`: Sampled token ids of shape `(...)`.
    """
    tiny = torch.finfo(uniform.dtype).tiny
    epsilon = torch.finfo(uniform.dtype).eps
    gumbel = -torch.log(-torch.log(uniform.clamp(min=tiny, max=1.0 - epsilon)))
    return torch.argmax(log_probs + gumbel, dim=-1)


def entropy_bound_acceptance(token_entropy: torch.Tensor, entropy_bound: float) -> torch.Tensor:
    """
    Selects the lowest-entropy positions whose summed entropy, excluding the largest, stays within the bound.

    Args:
        token_entropy (`torch.Tensor` of shape `(batch_size, canvas_length)`):
            Entropy of each canvas position.
        entropy_bound (`float`):
            Upper bound on the joint entropy of the accepted positions.

    Returns:
        `torch.Tensor`: Boolean mask of accepted positions with shape `(batch_size, canvas_length)`.
    """
    sorted_entropy, sorted_indices = torch.sort(token_entropy, dim=-1, descending=False)
    cumulative_entropy = torch.cumsum(sorted_entropy, dim=-1)
    sorted_accepted = cumulative_entropy - sorted_entropy <= entropy_bound
    return torch.scatter(torch.zeros_like(sorted_accepted), dim=-1, index=sorted_indices, src=sorted_accepted)


def finalize_canvas(
    canvas: torch.Tensor,
    finished_sequences: torch.Tensor,
    eos_token_ids: torch.Tensor,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pads everything after the first end-of-sequence token and the whole canvas of already finished sequences.

    Args:
        canvas (`torch.Tensor` of shape `(batch_size, canvas_length)`):
            Denoised canvas.
        finished_sequences (`torch.Tensor` of shape `(batch_size,)`):
            Sequences that ended in an earlier canvas.
        eos_token_ids (`torch.Tensor` of shape `(num_eos,)`):
            End-of-sequence token ids.
        pad_token_id (`int`):
            Padding token id.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: The padded canvas and the updated finished-sequence mask.
    """
    is_eos = torch.isin(canvas, eos_token_ids)
    eos_count = is_eos.cumsum(dim=-1)
    after_first_eos = (eos_count > 0) & ~((eos_count == 1) & is_eos)
    pad_mask = after_first_eos | finished_sequences[:, None]
    padded_canvas = torch.where(pad_mask, torch.full_like(canvas, pad_token_id), canvas)
    return padded_canvas, finished_sequences | is_eos.any(dim=-1)
