from torch_mojo_backend.torch_compile_backend.utils import get_accelerators


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
        _enable_apple_fast_add()
