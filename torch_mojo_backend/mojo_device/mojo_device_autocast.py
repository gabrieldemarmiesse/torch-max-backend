"""AutocastPrivateUse1 policies for eager Mojo tensors.

PyTorch lets a private backend advertise supported AMP dtypes, but it does not
install any operator policies for that backend. These wrappers mirror the CUDA
policies needed by nanoGPT: matmul/attention run in the active lower precision,
while normalization and NLL run in FP32. Unlisted operations fall through.
"""

import functools
from collections.abc import Mapping
from typing import no_type_check

import torch

_registered = False
_fallback_library = None
_aten_library = None

_AUTOCAST_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.AutocastPrivateUse1)


@no_type_check
def _cast_mojo_floating(value, dtype):
    """Recursively cast eligible Mojo floating tensors, leaving metadata alone."""
    if isinstance(value, torch.Tensor):
        if (
            value.device.type == "mojo"
            and value.is_floating_point()
            and value.dtype != torch.float64
            and value.dtype != dtype
        ):
            return value.to(dtype=dtype)
        return value
    if isinstance(value, tuple):
        return tuple(_cast_mojo_floating(item, dtype) for item in value)
    if isinstance(value, list):
        return [_cast_mojo_floating(item, dtype) for item in value]
    if isinstance(value, Mapping):
        return {key: _cast_mojo_floating(item, dtype) for key, item in value.items()}
    return value


def _policy_wrapper(op, dtype_getter):
    @functools.wraps(op)
    @no_type_check
    def wrapper(*args, **kwargs):
        # Casts themselves must redispatch below AutocastPrivateUse1, otherwise
        # their internal _to_copy calls would re-enter this policy layer.
        with torch._C._ExcludeDispatchKeyGuard(_AUTOCAST_KEYSET):
            dtype = dtype_getter()
            return op(
                *_cast_mojo_floating(args, dtype), **_cast_mojo_floating(kwargs, dtype)
            )

    return wrapper


def _lower_precision_wrapper(op):
    return _policy_wrapper(op, lambda: torch.get_autocast_dtype("mojo"))


def _fp32_wrapper(op):
    return _policy_wrapper(op, lambda: torch.float32)


def register_autocast_ops() -> None:
    """Install fallthrough plus the explicit CUDA-matching GPT policies."""
    global _registered, _fallback_library, _aten_library
    if _registered:
        return

    _fallback_library = torch.library.Library("_", "IMPL", "AutocastPrivateUse1")
    _fallback_library.fallback(torch.library.fallthrough_kernel)
    _aten_library = torch.library.Library("aten", "IMPL", "AutocastPrivateUse1")

    lower_precision_ops = (
        torch.ops.aten.addmm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.linear.default,
        torch.ops.aten.matmul.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.scaled_dot_product_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
    )
    fp32_ops = (
        torch.ops.aten.layer_norm.default,
        torch.ops.aten.native_layer_norm.default,
        torch.ops.aten.nll_loss.default,
        torch.ops.aten.nll_loss_forward.default,
    )
    for op in lower_precision_ops:
        _aten_library.impl(op._schema.name, _lower_precision_wrapper(op))
    for op in fp32_ops:
        _aten_library.impl(op._schema.name, _fp32_wrapper(op))

    _registered = True


__all__ = ["register_autocast_ops"]
