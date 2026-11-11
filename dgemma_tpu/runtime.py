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
