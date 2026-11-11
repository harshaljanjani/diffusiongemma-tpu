import torch
from torch._dynamo.backends.registry import lookup_backend


class HostSyncCounter:
    """Context manager counting tensor operations that force a device-to-host synchronization."""

    _SYNCING_METHODS = ("__bool__", "item", "tolist", "cpu", "numpy", "__index__", "__float__", "__int__")

    def __init__(self):
        self.count = 0
        self._originals = {}

    def __enter__(self):
        for name in self._SYNCING_METHODS:
            original = getattr(torch.Tensor, name)
            self._originals[name] = original
            setattr(torch.Tensor, name, self._counting(original))
        return self

    def __exit__(self, *exc_info):
        for name, original in self._originals.items():
            setattr(torch.Tensor, name, original)

    def _counting(self, original):
        def wrapper(tensor, *args, **kwargs):
            self.count += 1
            return original(tensor, *args, **kwargs)

        return wrapper


class GraphCountingBackend:
    """
    `torch.compile` backend that counts compiled graphs before delegating to another backend.

    Args:
        inner_backend (`str`, *optional*, defaults to `"aot_eager"`):
            Name of the backend that compiles each graph.
    """

    def __init__(self, inner_backend: str = "aot_eager"):
        self.inner_backend = inner_backend
        self.graph_sizes = []

    @property
    def num_graphs(self) -> int:
        """
        Returns:
            `int`: Number of graphs compiled so far.
        """
        return len(self.graph_sizes)

    def __call__(self, graph_module, example_inputs):
        self.graph_sizes.append(len(graph_module.graph.nodes))
        return lookup_backend(self.inner_backend)(graph_module, example_inputs)
