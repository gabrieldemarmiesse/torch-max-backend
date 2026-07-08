"""ATen op registrations for max_device eager mode.

Every op is either bound to its fast implementation in
`eager_kernels/aten_fast.py` (Mojo kernels over raw pointers) or raises
`NotImplementedError` with an actionable message. The old graph fallback
(per-op MAX graph build + interpret, ~2.2 ms/op) is gone — see
docs/strided_owning_tensors_design.md.
"""

from collections.abc import Callable
from typing import no_type_check

import torch
from max.experimental.torch.torch import torch_dtype_to_max

from torch_max_backend.max_device.torch_max_tensor import (
    TorchMaxTensor,
    _copy_strided_into,
    _rebind_payload,
    find_equivalent_max_device,
)

# Global registry for functions to register
_aten_ops_registry: list[tuple[str, Callable]] = []


def register_aten_op(op_name: str):
    """Decorator to mark a function for aten op registration.

    Args:
        op_name: The aten operation name (e.g., "aten::add.Tensor")
    """

    def decorator(func: Callable) -> Callable:
        _aten_ops_registry.append((op_name, func))
        return func

    return decorator


_aten_fast_module = None


def _fast():
    """The aten_fast module.

    Imported lazily: the first import triggers the (cached) Mojo kernel
    compilation, which pure torch.compile workloads should never pay for.
    """
    global _aten_fast_module
    if _aten_fast_module is None:
        from torch_max_backend.eager_kernels import aten_fast

        _aten_fast_module = aten_fast
    return _aten_fast_module


def _describe_args(args, kwargs) -> str:
    descs = []
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, TorchMaxTensor):
            descs.append(f"{tuple(a._shape)}:{a._dtype}")
        elif isinstance(a, torch.Tensor):
            descs.append(f"{tuple(a.shape)}:{a.dtype}:{a.device}")
    return ", ".join(descs) or "none"


def _unsupported(op_name: str, args=(), kwargs=None) -> NotImplementedError:
    return NotImplementedError(
        f"{op_name} is not supported by max_device eager mode for these inputs "
        f"(tensor args: {_describe_args(args, kwargs or {})}). The graph "
        "fallback was removed; add a fast kernel in "
        "torch_max_backend/eager_kernels/ or open an issue."
    )


def _eager_impl(fast_name: str, op_name: str) -> Callable:
    """Bind an op to its aten_fast implementation; raise on NOT_HANDLED."""
    fast_fn: Callable | None = None
    not_handled = None

    def dispatcher(*args, **kwargs):
        nonlocal fast_fn, not_handled
        if fast_fn is None:
            aten_fast = _fast()
            fast_fn = getattr(aten_fast, fast_name)
            not_handled = aten_fast.NOT_HANDLED
        result = fast_fn(*args, **kwargs)
        if result is not_handled:
            raise _unsupported(op_name, args, kwargs)
        return result

    return dispatcher


def _not_implemented(op_name: str) -> Callable:
    """Explicit raiser for ops that used to run through the graph fallback
    and have no fast implementation yet. Registered (instead of left
    unregistered) so users get an actionable message, and so the remaining
    surface is greppable."""

    def raiser(*args, **kwargs):
        raise _unsupported(op_name, args, kwargs)

    return raiser


def _register_fast(op_name: str, fast_name: str):
    register_aten_op(op_name)(_eager_impl(fast_name, op_name))


def _register_missing(op_name: str):
    register_aten_op(op_name)(_not_implemented(op_name))


@no_type_check
def _copy_into_tensor(dst: TorchMaxTensor, src: TorchMaxTensor) -> None:
    """dst[...] = src[...] with dtype cast + broadcast, any strides."""
    aten_fast = _fast()
    if src._dtype != dst._dtype:
        src = aten_fast._cast_tensor(src, dst._dtype)
    if tuple(src._shape) != tuple(dst._shape):
        expanded = aten_fast.fast_aten_expand(src, dst._shape)
        if expanded is aten_fast.NOT_HANDLED:
            raise _unsupported("aten::copy_ (broadcast)", (dst, src))
        src = expanded
    aten_fast._copy_into(dst, src)


@no_type_check
def _out_variant(op_name: str, fast_name: str):
    """Wrap a functional fast implementation as an out= variant: compute,
    then copy into `out` (strided-safe)."""

    def dispatcher(*args, out: TorchMaxTensor, **kwargs):
        aten_fast = _fast()
        result = getattr(aten_fast, fast_name)(*args, **kwargs)
        if result is aten_fast.NOT_HANDLED:
            raise _unsupported(op_name, args, kwargs)
        if tuple(result._shape) == tuple(out._shape):
            _copy_into_tensor(out, result)
        else:
            # torch resizes mismatched out= tensors; rebind the payload.
            _rebind_payload(out, result)
        return out

    return dispatcher


# ----------------------------------------------------------------------------------
# Data transfer: H2D / D2H / D2D and dtype/device copies
# ----------------------------------------------------------------------------------


@register_aten_op("aten::_copy_from")
@no_type_check
def max_device__copy_from(self, dest):
    src_is_max = isinstance(self, TorchMaxTensor)
    dest_is_max = isinstance(dest, TorchMaxTensor)

    if src_is_max and dest_is_max:
        if self._device != dest._device:
            # Cross max-device: bounce through the host.
            bounced = TorchMaxTensor._from_cpu(self._to_cpu_tensor(), dest._device)
            _copy_into_tensor(dest, bounced)
            return dest
        _copy_into_tensor(dest, self)
        return dest

    if src_is_max and not dest_is_max:
        # D2H; dest is a CPU torch tensor (copy_ handles cast/layout).
        dest.copy_(self._to_cpu_tensor())
        return dest

    if not src_is_max and dest_is_max:
        # H2D. Resolve dtype/broadcast on the host, then one upload.
        cpu = self.detach()
        torch_dtype = max_dtype_to_torch_dtype(dest._dtype)
        if cpu.dtype != torch_dtype:
            cpu = cpu.to(torch_dtype)
        if tuple(cpu.shape) != tuple(dest._shape):
            cpu = cpu.broadcast_to(dest._shape)
        cpu = cpu.contiguous()
        if dest._is_contiguous:
            if dest._numel > 0:
                from torch_max_backend import eager_kernels

                eager_kernels.tensor_holder.copy_from_host(
                    eager_kernels._ctx_ptr(dest._device),
                    dest._ptr,
                    cpu.data_ptr(),
                    dest._numel * dest._itemsize,
                )
        else:
            staged = TorchMaxTensor._from_cpu(cpu, dest._device)
            _copy_strided_into(dest, staged)
        return dest

    raise RuntimeError(
        f"invalid _copy_from configuration: {type(self)} -> {type(dest)}"
    )


@no_type_check
def max_dtype_to_torch_dtype(dtype):
    from max.experimental.torch import max_dtype_to_torch

    return max_dtype_to_torch(dtype)


@register_aten_op("aten::_to_copy")
@no_type_check
def max_device__to_copy(
    tensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
    non_blocking: bool = False,
    memory_format: torch.memory_format | None = None,
):
    aten_fast = _fast()
    if not isinstance(tensor, TorchMaxTensor):
        # CPU tensor moving onto max_device (optionally casting on host).
        t = tensor.detach()
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        return TorchMaxTensor._from_cpu(t, find_equivalent_max_device(device))

    result = tensor
    if dtype is not None:
        max_dtype = torch_dtype_to_max(dtype)
        if max_dtype != result._dtype:
            if (
                result._dtype in aten_fast._CAST_DTYPES
                and max_dtype in aten_fast._CAST_DTYPES
            ):
                result = aten_fast._cast_tensor(result, max_dtype)
            else:
                # Exotic dtype pair: cast on the host.
                cpu = result._to_cpu_tensor().to(dtype)
                if device is not None and device.type == "cpu":
                    return cpu
                target = (
                    find_equivalent_max_device(device)
                    if device is not None
                    else result._device
                )
                return TorchMaxTensor._from_cpu(cpu, target)
    if device is not None and device.type == "cpu":
        return result._to_cpu_tensor()
    if device is not None:
        target = find_equivalent_max_device(device)
        if target != result._device:
            return TorchMaxTensor._from_cpu(result._to_cpu_tensor(), target)
    if result is tensor:
        # _to_copy always returns a fresh tensor.
        result = tensor._materialize_contiguous()
    return result


# ----------------------------------------------------------------------------------
# Factories
# ----------------------------------------------------------------------------------


@register_aten_op("aten::empty.memory_format")
@no_type_check
def max_device_empty_memory_format(
    size, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
) -> TorchMaxTensor:
    dtype = torch.get_default_dtype() if dtype is None else dtype
    return TorchMaxTensor._alloc(
        tuple(size), torch_dtype_to_max(dtype), find_equivalent_max_device(device)
    )


@register_aten_op("aten::empty_strided.memory_format")
@register_aten_op("aten::empty_strided")
@no_type_check
def empty_strided(
    size, stride, *, dtype=None, layout=None, device=None, pin_memory=None
) -> TorchMaxTensor:
    # The requested strides are ignored: allocation is always contiguous
    # (matching the previous behavior; our metadata is self-consistent).
    dtype = torch.get_default_dtype() if dtype is None else dtype
    return TorchMaxTensor._alloc(
        tuple(size), torch_dtype_to_max(dtype), find_equivalent_max_device(device)
    )


@register_aten_op("aten::empty_permuted")
@no_type_check
def max_device_empty_permuted(
    size, physical_layout, *, dtype=None, layout=None, device=None, pin_memory=None
) -> TorchMaxTensor:
    # Uninitialized memory: a contiguous allocation of `size` is valid.
    dtype = torch.get_default_dtype() if dtype is None else dtype
    return TorchMaxTensor._alloc(
        tuple(size), torch_dtype_to_max(dtype), find_equivalent_max_device(device)
    )


@register_aten_op("aten::empty_like")
@no_type_check
def max_device_empty_like(
    self: TorchMaxTensor,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
) -> TorchMaxTensor:
    max_dtype = self._dtype if dtype is None else torch_dtype_to_max(dtype)
    max_device = self._device if device is None else find_equivalent_max_device(device)
    return TorchMaxTensor._alloc(self._shape, max_dtype, max_device)


@no_type_check
def _fast_filled_tensor(size, value, dtype, device):
    """Filled-tensor factory (alloc + Fill), or raises."""
    try:
        max_dtype = torch_dtype_to_max(dtype)
    except (KeyError, ValueError):
        raise _unsupported("aten::full (dtype)", (dtype,)) from None
    result = _fast().fast_filled(
        size, value, max_dtype, find_equivalent_max_device(device)
    )
    if result is None:
        raise _unsupported("aten::full", (size, value, dtype))
    return result


@register_aten_op("aten::full")
@no_type_check
def max_device_full(
    size, fill_value, *, dtype=None, layout=None, device=None, pin_memory=None
) -> TorchMaxTensor:
    if dtype is not None:
        resolved = dtype
    elif isinstance(fill_value, bool):
        resolved = torch.bool
    elif isinstance(fill_value, int):
        resolved = torch.int64
    else:
        resolved = torch.get_default_dtype()
    return _fast_filled_tensor(size, fill_value, resolved, device)


@register_aten_op("aten::full_like")
@no_type_check
def max_device_full_like(
    self: TorchMaxTensor,
    fill_value,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
) -> TorchMaxTensor:
    max_dtype = self._dtype if dtype is None else torch_dtype_to_max(dtype)
    max_device = self._device if device is None else find_equivalent_max_device(device)
    result = _fast().fast_filled(self._shape, fill_value, max_dtype, max_device)
    if result is None:
        raise _unsupported("aten::full_like", (self, fill_value))
    return result


@register_aten_op("aten::ones")
@no_type_check
def max_device_ones(
    size, *, dtype=None, layout=None, device=None, pin_memory=None
) -> TorchMaxTensor:
    resolved = torch.get_default_dtype() if dtype is None else dtype
    return _fast_filled_tensor(size, 1, resolved, device)


@register_aten_op("aten::ones_like")
@no_type_check
def max_device_ones_like(
    self: TorchMaxTensor,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
) -> TorchMaxTensor:
    return max_device_full_like(self, 1, dtype=dtype, device=device)


@register_aten_op("aten::zeros")
@no_type_check
def max_device_zeros(
    size, *, dtype=None, layout=None, device=None, pin_memory=None
) -> TorchMaxTensor:
    resolved = torch.get_default_dtype() if dtype is None else dtype
    return _fast_filled_tensor(size, 0, resolved, device)


@register_aten_op("aten::zeros_like")
@no_type_check
def max_device_zeros_like(
    self: TorchMaxTensor,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
) -> TorchMaxTensor:
    return max_device_full_like(self, 0, dtype=dtype, device=device)


@register_aten_op("aten::scalar_tensor")
@no_type_check
def max_device_scalar_tensor(
    s, dtype=None, layout=None, device=None, pin_memory=None
) -> TorchMaxTensor:
    resolved = torch.float32 if dtype is None else dtype
    return _fast_filled_tensor((), s, resolved, device)


@no_type_check
def _host_arange_tensor(start, end, step, dtype: torch.dtype | None) -> torch.Tensor:
    """torch.arange built on the host (exact torch semantics)."""
    return torch.arange(start, end, step, dtype=dtype)


@register_aten_op("aten::arange")
@no_type_check
def max_device_arange(
    start, end=None, step=1, *, dtype=None, layout=None, device=None, pin_memory=None
) -> TorchMaxTensor:
    # Build on the host with exact torch semantics, then one H2D copy.
    if end is None:
        start, end = 0, start
    cpu = _host_arange_tensor(start, end, step, dtype)
    return TorchMaxTensor._from_cpu(cpu, find_equivalent_max_device(device))


@register_aten_op("aten::arange.start_out")
@no_type_check
def max_device_arange_start_out(start, end, step=1, *, out) -> TorchMaxTensor:
    # torch.arange(start, end, step, device=...) dispatches to the out
    # variant with a pre-allocated `out` of the right size and dtype.
    cpu = _host_arange_tensor(start, end, step, max_dtype_to_torch_dtype(out._dtype))
    staged = TorchMaxTensor._from_cpu(cpu, out._device)
    if tuple(staged._shape) == tuple(out._shape):
        _copy_into_tensor(out, staged)
    else:
        _rebind_payload(out, staged)
    return out


@register_aten_op("aten::normal_")
@no_type_check
def max_device_normal_(
    self: TorchMaxTensor, mean: float = 0.0, std: float = 1.0, generator=None
) -> TorchMaxTensor:
    if generator is not None:
        raise _unsupported("aten::normal_ (generator)", (self,))
    cpu = torch.empty(self._shape, dtype=max_dtype_to_torch_dtype(self._dtype)).normal_(
        mean, std
    )
    staged = TorchMaxTensor._from_cpu(cpu, self._device)
    _copy_into_tensor(self, staged)
    return self


# ----------------------------------------------------------------------------------
# In-place ops with custom plumbing
# ----------------------------------------------------------------------------------


@register_aten_op("aten::add_.Tensor")
@no_type_check
def max_device_add_(self: TorchMaxTensor, other, alpha: float = 1.0) -> TorchMaxTensor:
    result = _fast().fast_aten_add_(self, other, alpha)
    if result is None:
        raise _unsupported("aten::add_.Tensor", (self, other))
    return result


@register_aten_op("aten::fill_.Scalar")
@no_type_check
def max_device_fill__scalar(self: TorchMaxTensor, value) -> TorchMaxTensor:
    result = _fast().fast_aten_fill__scalar(self, value)
    if result is None:
        raise _unsupported("aten::fill_.Scalar", (self, value))
    return result


@register_aten_op("aten::masked_fill_.Scalar")
@register_aten_op("aten::masked_fill_.Tensor")
@no_type_check
def max_device_masked_fill_(
    self: TorchMaxTensor, mask: TorchMaxTensor, value
) -> TorchMaxTensor:
    result = _fast().fast_aten_masked_fill_(self, mask, value)
    if result is None:
        raise _unsupported("aten::masked_fill_", (self, mask, value))
    return result


@register_aten_op("aten::relu_")
@no_type_check
def max_device_relu_(self: TorchMaxTensor) -> TorchMaxTensor:
    aten_fast = _fast()
    result = aten_fast.fast_aten_relu(self)
    if result is aten_fast.NOT_HANDLED:
        raise _unsupported("aten::relu_", (self,))
    _copy_into_tensor(self, result)
    return self


@register_aten_op("aten::zero_")
@no_type_check
def max_device_zero_(self: TorchMaxTensor) -> TorchMaxTensor:
    return max_device_fill__scalar(self, 0)


# ----------------------------------------------------------------------------------
# Out-variants
# ----------------------------------------------------------------------------------

register_aten_op("aten::mul.out")(_out_variant("aten::mul.out", "fast_aten_mul"))
register_aten_op("aten::mean.out")(_out_variant("aten::mean.out", "fast_aten_mean"))
register_aten_op("aten::any.out")(_out_variant("aten::any.out", "fast_aten_any"))
register_aten_op("aten::isin.Tensor_Tensor_out")(
    _out_variant("aten::isin.Tensor_Tensor_out", "fast_aten_isin")
)


@register_aten_op("aten::min.dim_min")
@no_type_check
def max_device_min_dim_min(
    input: TorchMaxTensor,
    dim: int,
    keepdim: bool = False,
    min: TorchMaxTensor | None = None,
    min_indices: TorchMaxTensor | None = None,
) -> tuple[TorchMaxTensor, TorchMaxTensor]:
    """Out-variant of torch.min along a dim: writes values into `min` and
    int64 indices into `min_indices` (resizing via payload rebind when the
    pre-allocated shapes don't match, like the other out-variants)."""
    aten_fast = _fast()
    result = aten_fast.fast_aten_min_dim(input, dim, keepdim)
    if result is aten_fast.NOT_HANDLED:
        raise _unsupported("aten::min.dim_min", (input,))
    values, indices = result
    for dst, src in ((min, values), (min_indices, indices)):
        if dst is None:
            continue
        if tuple(dst._shape) == tuple(src._shape):
            _copy_into_tensor(dst, src)
        else:
            _rebind_payload(dst, src)
    return (min, min_indices)


# ----------------------------------------------------------------------------------
# Fast-implemented ops (alphabetical)
# ----------------------------------------------------------------------------------

_register_fast("aten::_local_scalar_dense", "fast_aten__local_scalar_dense")
_register_fast("aten::_log_softmax", "fast_aten__log_softmax")
_register_fast(
    "aten::_native_batch_norm_legit_no_training",
    "fast_aten__native_batch_norm_legit_no_training",
)
_register_fast("aten::_softmax", "fast_aten__softmax")
_register_fast("aten::_unsafe_view", "fast_aten__unsafe_view")
_register_fast("aten::add.Tensor", "fast_aten_add")
_register_fast("aten::addmm", "fast_aten_addmm")
_register_fast("aten::alias", "fast_aten_alias")
_register_fast("aten::all", "fast_aten_all")
_register_fast("aten::all.dim", "fast_aten_all")
_register_fast("aten::all.dims", "fast_aten_all")
_register_fast("aten::amax", "fast_aten_amax")
_register_fast("aten::amin", "fast_aten_amin")
_register_fast("aten::any", "fast_aten_any")
_register_fast("aten::any.dim", "fast_aten_any")
_register_fast("aten::any.dims", "fast_aten_any")
_register_fast("aten::argmax", "fast_aten_argmax")
_register_fast("aten::argmin", "fast_aten_argmin")
_register_fast("aten::bitwise_and.Scalar", "fast_aten_bitwise_and")
_register_fast("aten::bitwise_and.Tensor", "fast_aten_bitwise_and")
_register_fast("aten::bitwise_not", "fast_aten_bitwise_not")
_register_fast("aten::bitwise_or.Scalar", "fast_aten_bitwise_or")
_register_fast("aten::bitwise_or.Tensor", "fast_aten_bitwise_or")
_register_fast("aten::bitwise_xor.Scalar", "fast_aten_bitwise_xor")
_register_fast("aten::bitwise_xor.Tensor", "fast_aten_bitwise_xor")
_register_fast("aten::bmm", "fast_aten_bmm")
_register_fast("aten::cat", "fast_aten_cat")
_register_fast("aten::clone", "fast_aten_clone")
_register_fast("aten::convolution", "fast_aten_convolution")
_register_fast("aten::cumsum", "fast_aten_cumsum")
_register_fast("aten::detach", "fast_aten_detach")
_register_fast("aten::div.Tensor", "fast_aten_div")
_register_fast("aten::embedding", "fast_aten_embedding")
_register_fast("aten::eq", "fast_aten_eq")
_register_fast("aten::eq.Scalar", "fast_aten_eq")
_register_fast("aten::eq.Tensor", "fast_aten_eq")
_register_fast("aten::exp", "fast_aten_exp")
_register_fast("aten::expand", "fast_aten_expand")
_register_fast("aten::fill.Scalar", "fast_aten_fill_scalar")
_register_fast("aten::ge", "fast_aten_ge")
_register_fast("aten::ge.Scalar", "fast_aten_ge")
_register_fast("aten::ge.Tensor", "fast_aten_ge")
_register_fast("aten::gt", "fast_aten_gt")
_register_fast("aten::gt.Scalar", "fast_aten_gt")
_register_fast("aten::gt.Tensor", "fast_aten_gt")
_register_fast("aten::isin.Tensor_Tensor", "fast_aten_isin")
_register_fast("aten::le", "fast_aten_le")
_register_fast("aten::le.Scalar", "fast_aten_le")
_register_fast("aten::le.Tensor", "fast_aten_le")
_register_fast("aten::linear", "fast_aten_linear")
_register_fast("aten::lt", "fast_aten_lt")
_register_fast("aten::lt.Scalar", "fast_aten_lt")
_register_fast("aten::lt.Tensor", "fast_aten_lt")
_register_fast("aten::masked_fill.Scalar", "fast_aten_masked_fill")
_register_fast("aten::masked_fill.Tensor", "fast_aten_masked_fill")
_register_fast("aten::max", "fast_aten_max")
_register_fast("aten::max_pool2d_with_indices", "fast_aten_max_pool2d_with_indices")
_register_fast("aten::maximum", "fast_aten_maximum")
_register_fast("aten::mean", "fast_aten_mean")
# Registering the base name only covers the default overload; mean.dim would
# otherwise get decomposed by PyTorch into a chain of sum/div/... ops.
_register_fast("aten::mean.dim", "fast_aten_mean")
_register_fast("aten::min", "fast_aten_min")
_register_fast("aten::minimum", "fast_aten_minimum")
_register_fast("aten::mm", "fast_aten_mm")
_register_fast("aten::mul.Tensor", "fast_aten_mul")
_register_fast("aten::native_batch_norm", "fast_aten_native_batch_norm")
_register_fast("aten::native_layer_norm", "fast_aten_native_layer_norm")
_register_fast("aten::ne", "fast_aten_ne")
_register_fast("aten::ne.Scalar", "fast_aten_ne")
_register_fast("aten::ne.Tensor", "fast_aten_ne")
_register_fast("aten::permute", "fast_aten_permute")
_register_fast("aten::pow.Tensor_Scalar", "fast_aten_pow")
_register_fast("aten::relu", "fast_aten_relu")
_register_fast(
    "aten::scaled_dot_product_attention", "fast_aten_scaled_dot_product_attention"
)
_register_fast("aten::select.int", "fast_aten_select")
_register_fast("aten::slice.Tensor", "fast_aten_slice")
_register_fast("aten::softmax.int", "fast_aten_softmax")
_register_fast("aten::split.Tensor", "fast_aten_split")
_register_fast("aten::split_with_sizes", "fast_aten_split_with_sizes")
_register_fast("aten::squeeze.dim", "fast_aten_squeeze_dim")
_register_fast("aten::sub.Tensor", "fast_aten_sub")
_register_fast("aten::sum.dim_IntList", "fast_aten_sum")
_register_fast("aten::t", "fast_aten_t")
_register_fast("aten::tanh", "fast_aten_tanh")
_register_fast("aten::transpose.int", "fast_aten_transpose")
_register_fast("aten::unbind.int", "fast_aten_unbind")
_register_fast("aten::unsqueeze", "fast_aten_unsqueeze")
_register_fast("aten::var.correction", "fast_aten_var")
_register_fast("aten::view", "fast_aten_view")
_register_fast("aten::where.self", "fast_aten_where")


# ----------------------------------------------------------------------------------
# Ops with no fast implementation yet: explicit raisers (previously served
# by the graph fallback). Implement in eager_kernels and move up.
# ----------------------------------------------------------------------------------

_register_missing("aten::_adaptive_avg_pool2d")
_register_missing("aten::_adaptive_avg_pool2d_backward")
_register_missing("aten::_scaled_dot_product_attention_math")
_register_missing("aten::_scaled_dot_product_efficient_attention")
_register_missing("aten::_scaled_dot_product_flash_attention")
_register_missing("aten::abs")
_register_missing("aten::acos")
_register_missing("aten::addcdiv")
_register_missing("aten::addcmul")
_register_missing("aten::asinh")
_register_missing("aten::atanh")
_register_missing("aten::avg_pool2d")
_register_missing("aten::ceil")
_register_missing("aten::clamp")
_register_missing("aten::cos")
_register_missing("aten::cosh")
_register_missing("aten::erf")
_register_missing("aten::floor")
_register_missing("aten::floordiv")
_register_missing("aten::gelu")
_register_missing("aten::gelu_backward")
_register_missing("aten::index.Tensor")
_register_missing("aten::isnan")
_register_missing("aten::log")
_register_missing("aten::log1p")
_register_missing("aten::logical_and")
_register_missing("aten::logical_not")
_register_missing("aten::logical_xor")
_register_missing("aten::native_group_norm")
_register_missing("aten::neg")
_register_missing("aten::nonzero")
_register_missing("aten::pow.Tensor_Tensor")
_register_missing("aten::reciprocal")
_register_missing("aten::remainder")
_register_missing("aten::repeat")
_register_missing("aten::rsqrt")
_register_missing("aten::scatter.src")
_register_missing("aten::scatter.value")
_register_missing("aten::select_scatter")
_register_missing("aten::sigmoid")
_register_missing("aten::sign")
_register_missing("aten::silu")
_register_missing("aten::sin")
_register_missing("aten::sinh")
_register_missing("aten::sqrt")
_register_missing("aten::stack")
_register_missing("aten::tan")
_register_missing("aten::tril")
_register_missing("aten::triu")
_register_missing("aten::upsample_bilinear2d")
