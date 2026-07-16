from functools import wraps

import torch

from torch_mojo_backend.torch_compile_backend.utils import get_accelerators

from .torch_mojo_tensor import TorchMojoTensor

_PATCH_MARKER = "_torch_mojo_backend_fused_mojo_gelu"


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


def _enable_apple_fast_add() -> None:
    """Select the isolated Metal contiguous-add implementation.

    Patching once during device registration keeps CUDA and ROCm on the
    original Python and Mojo paths with no per-call target check.
    """
    from torch_mojo_backend.eager_kernels import aten_fast

    aten_fast.fast_aten_add = aten_fast.fast_aten_add_apple


def register_apple_optimizations() -> None:
    """Install optional integrations that are profitable only on Apple GPUs."""
    if any(device.api == "metal" for device in get_accelerators()):
        _patch_transformers_new_gelu()
        _enable_apple_fast_add()
