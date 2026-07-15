from functools import wraps

import torch

from torch_max_backend.torch_compile_backend.utils import get_accelerators

from .torch_max_tensor import TorchMojoTensor

_PATCH_MARKER = "_torch_max_backend_fused_mojo_gelu"


def _patch_transformers_new_gelu() -> bool:
    """Use the fused ATen tanh-GELU for Transformers tensors on mojo.

    Hugging Face's ``NewGELUActivation`` spells the approximation as seven
    eager PyTorch operations. ``aten::gelu(..., approximate='tanh')`` uses
    the same formula, but reaches our existing single-launch GELU kernel.
    Keep the original forward for every non-mojo tensor so registering the
    backend cannot change CPU, CUDA, or MPS model execution.
    """
    try:
        from transformers.activations import NewGELUActivation
    except ImportError:
        return False

    original_forward = NewGELUActivation.forward
    if getattr(original_forward, _PATCH_MARKER, False):
        return False

    @wraps(original_forward)
    def fused_mojo_forward(self: torch.nn.Module, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, TorchMojoTensor):
            return torch.nn.functional.gelu(input, approximate="tanh")
        return original_forward(self, input)

    setattr(fused_mojo_forward, _PATCH_MARKER, True)
    NewGELUActivation.forward = fused_mojo_forward
    return True


def register_apple_optimizations() -> None:
    """Install optional integrations that are profitable only on Apple GPUs."""
    if any(device.api == "metal" for device in get_accelerators()):
        _patch_transformers_new_gelu()
