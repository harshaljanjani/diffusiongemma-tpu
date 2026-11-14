import os

import torch
import torch.distributed as dist


def is_distributed_launch() -> bool:
    """
    Returns:
        `bool`: Whether this process was started as one rank of a multi-process launch.
    """
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def initialize_process_group(device_type: str) -> tuple[int, int]:
    """
    Joins the process group of a `torchrun` launch.

    Args:
        device_type (`str`):
            Device type of the model, which selects the communication backend.

    Returns:
        `tuple[int, int]`: Rank of this process and the number of ranks.
    """
    if not dist.is_initialized():
        backend = "tpu_dist" if device_type == "tpu" else "gloo"
        dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size()


def shard_experts(model: torch.nn.Module, rank: int, world_size: int) -> int:
    """
    Keeps one slice of the intermediate dimension of every expert on this rank.

    Each expert computes `down(act(gate(x)) * up(x))`, a sum over intermediate units, so slicing those units across
    ranks and summing the partial outputs reproduces the full layer with a single all-reduce.

    Args:
        model (`torch.nn.Module`):
            Model holding full expert weights on the host.
        rank (`int`):
            Rank of this process.
        world_size (`int`):
            Number of ranks the experts are split across.

    Returns:
        `int`: Number of expert modules that were sharded.
    """
    num_sharded = 0
    for module in _expert_modules(model):
        if getattr(module, "_expert_shard", None) is not None:
            continue
        intermediate_size = module.down_proj.shape[-1]
        if intermediate_size % world_size != 0:
            raise ValueError(f"Expert intermediate size {intermediate_size} is not divisible by {world_size} ranks.")
        shard_size = intermediate_size // world_size
        start = rank * shard_size
        gate_rows = torch.arange(start, start + shard_size)
        rows = torch.cat([gate_rows, gate_rows + intermediate_size])
        module.gate_up_proj.data = module.gate_up_proj.data[:, rows, :].contiguous()
        module.down_proj.data = module.down_proj.data[:, :, start : start + shard_size].contiguous()
        module._expert_shard = (rank, world_size)
        module.forward = _all_reduced(module.forward)
        num_sharded += 1
    return num_sharded


def _expert_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    seen_weights = set()
    modules = []
    for module in model.modules():
        if hasattr(module, "gate_up_proj") and hasattr(module, "down_proj"):
            if id(module.gate_up_proj) not in seen_weights:
                seen_weights.add(id(module.gate_up_proj))
                modules.append(module)
            else:
                module._expert_shard = "shared"
                module.forward = _all_reduced(module.forward)
    return modules


def _all_reduced(forward):
    def reduced_forward(*args, **kwargs):
        output = forward(*args, **kwargs)
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output

    return reduced_forward


def spmd_decorator(device_type: str):
    """
    Returns the decorator that lets TorchTPU fuse collectives into a compiled graph, or `None` off TPU.

    Args:
        device_type (`str`):
            Device type of the model.

    Returns:
        `Callable` or `None`: `torch.tpu.distributed.spmd_safe` on TPU.
    """
    if device_type != "tpu":
        return None
    return torch.tpu.distributed.spmd_safe
