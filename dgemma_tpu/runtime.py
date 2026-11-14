import time
from contextlib import contextmanager

import torch


def resolve_device(name: str = "auto") -> torch.device:
    """
    Returns the requested device, preferring a TPU when `name` is `"auto"`.

    Args:
        name (`str`, *optional*, defaults to `"auto"`):
            Device name, or `"auto"` to pick the available accelerator and fall back to the CPU.

    Returns:
        `torch.device`: The resolved device.
    """
    if name != "auto":
        return torch.device(name)
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """
    Blocks until all queued work on an asynchronous device has finished.

    Args:
        device (`torch.device`):
            Device to wait for.
    """
    if device.type != "cpu":
        torch.accelerator.synchronize(device)


def peak_memory_bytes(device: torch.device) -> int | None:
    """
    Returns the peak device memory allocated by this process, when the backend reports it.

    Args:
        device (`torch.device`):
            Device to query.

    Returns:
        `int` or `None`: Peak allocated bytes.
    """
    if device.type == "cpu":
        return None
    try:
        return int(torch.accelerator.max_memory_allocated(device))
    except (RuntimeError, AttributeError, NotImplementedError):
        return None


@contextmanager
def timed(device: torch.device, timings: dict, name: str):
    """
    Measures the wall-clock time of a block after draining the device queue.

    Args:
        device (`torch.device`):
            Device the block runs on.
        timings (`dict`):
            Dictionary receiving the elapsed seconds under `name`.
        name (`str`):
            Key of the measurement.
    """
    synchronize(device)
    start = time.perf_counter()
    yield
    synchronize(device)
    timings[name] = time.perf_counter() - start


def compile_backend_for(device: torch.device, requested: str | None) -> str | None:
    """
    Maps a compile request to a `torch.compile` backend name.

    Args:
        device (`torch.device`):
            Device the model runs on.
        requested (`str`, *optional*):
            `"none"`, `"auto"` or an explicit backend name.

    Returns:
        `str` or `None`: Backend name, or `None` to run eagerly.
    """
    if requested in (None, "none"):
        return None
    if requested != "auto":
        return requested
    return "tpu" if device.type == "tpu" else "aot_eager"


def move_preserving_ties(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    Moves a model to a device while keeping parameters that are shared between modules shared.

    `Module.to` rebuilds each module's parameters on backends whose tensors cannot be swapped in place, which
    silently duplicates tied weights. Moving every distinct tensor once avoids both the copy and the extra memory.

    Args:
        model (`torch.nn.Module`):
            Model to move.
        device (`torch.device`):
            Target device.

    Returns:
        `torch.nn.Module`: The same model, with its parameters and buffers on `device`.
    """
    moved = {}
    for module in model.modules():
        for name, parameter in module._parameters.items():
            if parameter is None:
                continue
            if id(parameter) not in moved:
                new_parameter = torch.nn.Parameter(parameter.data.to(device), requires_grad=parameter.requires_grad)
                moved[id(parameter)] = (parameter, new_parameter)
            module._parameters[name] = moved[id(parameter)][1]
        for name, buffer in module._buffers.items():
            if buffer is None:
                continue
            if id(buffer) not in moved:
                moved[id(buffer)] = (buffer, buffer.to(device))
            module._buffers[name] = moved[id(buffer)][1]
    return model
