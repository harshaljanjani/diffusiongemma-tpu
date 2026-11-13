import torch
from transformers.integrations.moe import (
    ALL_EXPERTS_FUNCTIONS,
    batched_mm_experts_forward,
    grouped_mm_experts_forward,
)

DENSE_EXPERTS = "dense_all_experts"
DENSE_PREWEIGHTED_EXPERTS = "dense_preweighted"
TOKEN_COUNT_EXPERTS = "by_token_count"
BATCHED_MAX_TOKENS = 2
DENSE_MAX_TOKENS = 1536


def dense_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Runs every expert on every token with two batched matmuls and mixes the outputs with the routing weights.

    Trades redundant compute for a graph without sorts, gathers or per-step weight copies, which suits a canvas of
    a few hundred tokens where most experts are active anyway.

    Args:
        hidden_states (`torch.Tensor` of shape `(num_tokens, hidden_size)`):
            Token representations.
        top_k_index (`torch.Tensor` of shape `(num_tokens, top_k)`):
            Indices of the experts selected for each token.
        top_k_weights (`torch.Tensor` of shape `(num_tokens, top_k)`):
            Routing weight of each selected expert.

    Returns:
        `torch.Tensor`: Mixed expert outputs of shape `(num_tokens, hidden_size)`.
    """
    routing = _routing_matrix(self.num_experts, top_k_index, top_k_weights, hidden_states.dtype)
    gate_up = torch.matmul(self.gate_up_proj, hidden_states.transpose(0, 1))
    gate, up = gate_up.chunk(2, dim=1)
    activated = self.act_fn(gate) * up
    expert_outputs = torch.matmul(self.down_proj, activated)
    return torch.einsum("eht,te->th", expert_outputs, routing)


def dense_preweighted_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Variant of [`dense_experts_forward`] that applies the routing weights before the down projection.

    Args:
        hidden_states (`torch.Tensor` of shape `(num_tokens, hidden_size)`):
            Token representations.
        top_k_index (`torch.Tensor` of shape `(num_tokens, top_k)`):
            Indices of the experts selected for each token.
        top_k_weights (`torch.Tensor` of shape `(num_tokens, top_k)`):
            Routing weight of each selected expert.

    Returns:
        `torch.Tensor`: Mixed expert outputs of shape `(num_tokens, hidden_size)`.
    """
    routing = _routing_matrix(self.num_experts, top_k_index, top_k_weights, hidden_states.dtype)
    gate_up = torch.matmul(self.gate_up_proj, hidden_states.transpose(0, 1))
    gate, up = gate_up.chunk(2, dim=1)
    activated = self.act_fn(gate) * up * routing.transpose(0, 1)[:, None, :]
    return torch.einsum("ehi,eit->th", self.down_proj, activated)


def token_count_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Picks the fastest experts implementation on TPU v6e for the number of tokens in the pass.

    Gathering the selected expert weights wins for one or two tokens, running all experts densely wins up to
    roughly 1.5k tokens, and sorted grouped matmuls win beyond that.

    Args:
        hidden_states (`torch.Tensor` of shape `(num_tokens, hidden_size)`):
            Token representations.
        top_k_index (`torch.Tensor` of shape `(num_tokens, top_k)`):
            Indices of the experts selected for each token.
        top_k_weights (`torch.Tensor` of shape `(num_tokens, top_k)`):
            Routing weight of each selected expert.

    Returns:
        `torch.Tensor`: Mixed expert outputs of shape `(num_tokens, hidden_size)`.
    """
    num_tokens = hidden_states.shape[0]
    if num_tokens <= BATCHED_MAX_TOKENS:
        return batched_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights)
    if num_tokens <= DENSE_MAX_TOKENS:
        return dense_preweighted_experts_forward(self, hidden_states, top_k_index, top_k_weights)
    return grouped_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights)


def _routing_matrix(num_experts: int, top_k_index: torch.Tensor, top_k_weights: torch.Tensor, dtype: torch.dtype):
    routing = torch.zeros(top_k_index.shape[0], num_experts, dtype=top_k_weights.dtype, device=top_k_index.device)
    return routing.scatter(1, top_k_index, top_k_weights).to(dtype)


def register_experts_implementations() -> None:
    """Registers the dense expert implementations with the Transformers experts interface."""
    ALL_EXPERTS_FUNCTIONS[DENSE_EXPERTS] = dense_experts_forward
    ALL_EXPERTS_FUNCTIONS[DENSE_PREWEIGHTED_EXPERTS] = dense_preweighted_experts_forward
    ALL_EXPERTS_FUNCTIONS[TOKEN_COUNT_EXPERTS] = token_count_experts_forward


def set_experts_implementation(model: torch.nn.Module, name: str) -> None:
    """
    Selects the experts implementation used by every mixture-of-experts layer of a model.

    Args:
        model (`torch.nn.Module`):
            Model whose expert modules are switched.
        name (`str`):
            Registered implementation name, or `"eager"` for the reference loop.
    """
    register_experts_implementations()
    for module in model.modules():
        if hasattr(module, "gate_up_proj") and hasattr(module, "config"):
            module.config._experts_implementation = name
