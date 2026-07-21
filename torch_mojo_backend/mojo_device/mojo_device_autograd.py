"""AutogradPrivateUse1 plumbing for eager Mojo operations.

PyTorch's saved-variable plumbing returns a base ``torch.Tensor`` view of an
out-of-tree device tensor. That view keeps the version counter and saved-tensor
hooks working, but loses ``TorchMojoTensor``'s Python payload
(``_holder``/``_ptr``). These custom Functions save tensors normally, retain
only their non-Tensor payload metadata, and reattach that payload when backward
unpacks the saved variables.
"""

import math
from typing import no_type_check

import torch

from .torch_mojo_tensor import TorchMojoTensor

_registered = False


def _fast():
    from torch_mojo_backend.eager_kernels import aten_fast

    return aten_fast


@no_type_check
def _require_handled(result, operation: str) -> TorchMojoTensor:
    aten_fast = _fast()
    if result is None or result is aten_fast.NOT_HANDLED:
        raise NotImplementedError(
            f"{operation} is not supported by Mojo eager autograd for these inputs"
        )
    return result


@no_type_check
def _contiguous_view(tensor: TorchMojoTensor, shape) -> TorchMojoTensor:
    tensor = tensor._contig()
    return _require_handled(_fast().fast_aten_view(tensor, tuple(shape)), "view")


class _SavedMojoPayload:
    """Non-Tensor state needed to use a PyTorch-unpacked saved variable.

    ``SavedVariable::unpack`` deliberately returns a fresh base Tensor Python
    object for this out-of-tree backend. Its TensorImpl still carries the
    correct dispatch keys and shared version counter; restoring the subclass
    and Python-side allocation/layout fields makes that same object usable by
    eager Mojo kernels without retaining a Tensor directly on ``ctx``.

    Ordinarily the payload must retain the Mojo holder because TensorImpl does
    not own that out-of-tree allocation. With active saved-tensor hooks, the
    hook's packed object owns the saved value instead. In that case retaining
    the original holder here would defeat CPU/offload hooks, so ``holder`` is
    deliberately ``None`` and restore must use the unpack hook's value.
    """

    __slots__ = (
        "holder",
        "ptr",
        "shape",
        "strides",
        "offset",
        "dtype",
        "itemsize",
        "numel",
        "device",
        "torch_device",
        "is_contiguous",
    )

    @no_type_check
    def __init__(self, tensor: TorchMojoTensor):
        self.holder = None if _saved_tensor_hooks_active() else tensor._holder
        self.ptr = tensor._ptr
        self.shape = tensor._shape
        self.strides = tensor._strides
        self.offset = tensor._offset
        self.dtype = tensor._dtype
        self.itemsize = tensor._itemsize
        self.numel = tensor._numel
        self.device = tensor._device
        self.torch_device = tensor._torch_device
        self.is_contiguous = tensor._is_contiguous

    @no_type_check
    def restore(self, tensor: torch.Tensor) -> TorchMojoTensor:
        if isinstance(tensor, TorchMojoTensor) and hasattr(tensor, "_holder"):
            return self._validate_hook_result(tensor)

        # A saved-tensor hook may unpack to host memory. Honor the unpacked
        # value by moving it back rather than silently reading the original
        # Mojo allocation retained for the ordinary SavedVariable path.
        if tensor.device.type != "mojo":
            try:
                restored = tensor.to(self.torch_device)
            except Exception as exc:
                raise RuntimeError(
                    "saved-tensor hook result could not be restored to the "
                    f"original Mojo device {self.torch_device}"
                ) from exc
            return self._validate_hook_result(restored)

        if self.holder is None:
            raise RuntimeError(
                "saved-tensor hook returned an unusable Mojo tensor without "
                "a TorchMojoTensor allocation holder; its unpack hook must "
                "return a complete TorchMojoTensor or a host tensor"
            )

        tensor.__class__ = TorchMojoTensor
        tensor._holder = self.holder
        tensor._ptr = self.ptr
        tensor._shape = self.shape
        tensor._strides = self.strides
        tensor._offset = self.offset
        tensor._dtype = self.dtype
        tensor._itemsize = self.itemsize
        tensor._numel = self.numel
        tensor._device = self.device
        tensor._torch_device = self.torch_device
        tensor._is_contiguous = self.is_contiguous
        tensor.__dict__.pop("_spec", None)
        return tensor

    @no_type_check
    def _validate_hook_result(self, tensor: torch.Tensor) -> TorchMojoTensor:
        required = (
            "_holder",
            "_ptr",
            "_shape",
            "_strides",
            "_offset",
            "_dtype",
            "_itemsize",
            "_numel",
            "_device",
            "_torch_device",
            "_is_contiguous",
        )
        if not isinstance(tensor, TorchMojoTensor) or any(
            not hasattr(tensor, field) for field in required
        ):
            raise RuntimeError(
                "saved-tensor hook did not restore a complete TorchMojoTensor"
            )
        if (
            tuple(tensor._shape) != tuple(self.shape)
            or tensor._dtype != self.dtype
            or tensor._device != self.device
        ):
            raise RuntimeError(
                "saved-tensor hook restored incompatible Mojo tensor metadata: "
                f"expected shape={tuple(self.shape)}, dtype={self.dtype}, "
                f"device={self.torch_device}"
            )
        return tensor


def _saved_tensor_hooks_active() -> bool:
    """Best-effort active-hook detection across supported PyTorch versions.

    PyTorch does not expose this as public API. Newer versions accept a bool
    argument on ``_top_saved_tensors_default_hooks``; older variants accepted
    no argument. If neither private form is available, retaining the holder is
    the conservative compatibility fallback: correctness is preserved, only
    hook-driven allocation offload is less effective on that PyTorch version.
    """
    top_hooks = getattr(
        getattr(torch._C, "_autograd", None), "_top_saved_tensors_default_hooks", None
    )
    if top_hooks is None:
        return False
    try:
        return top_hooks(False) is not None
    except TypeError:
        try:
            return top_hooks() is not None
        except Exception:
            return False
    except Exception:
        return False


@no_type_check
def _restore_saved_mojo_tensors(ctx):
    saved_tensors = ctx.saved_tensors
    payloads = ctx.saved_payloads
    if len(saved_tensors) != len(payloads):
        raise RuntimeError("Mojo autograd saved-tensor payload count mismatch")
    return tuple(
        payload.restore(tensor)
        for tensor, payload in zip(saved_tensors, payloads, strict=True)
    )


class _LogSoftmaxAutograd(torch.autograd.Function):
    @staticmethod
    @no_type_check
    def forward(ctx, input, dim, half_to_float):
        output = _require_handled(
            _fast().fast_aten__log_softmax(input, dim, half_to_float),
            "aten::_log_softmax",
        )
        ctx.save_for_backward(output)
        ctx.saved_payloads = (_SavedMojoPayload(output),)
        ctx.dim = dim % len(input._shape) if input._shape else 0
        ctx.input_dtype = input._dtype
        return output

    @staticmethod
    @no_type_check
    def backward(ctx, grad_output):
        aten_fast = _fast()
        (output,) = _restore_saved_mojo_tensors(ctx)
        if not output._shape:
            return (
                aten_fast.fast_filled((), 0.0, ctx.input_dtype, output._device),
                None,
                None,
            )

        summed = _require_handled(
            aten_fast.fast_aten_sum(grad_output, dim=[ctx.dim], keepdim=True),
            "log_softmax backward reduction",
        )
        probabilities = _require_handled(
            aten_fast.fast_aten_exp(output), "log_softmax backward exp"
        )
        grad_input = _require_handled(
            aten_fast.fast_aten_addcmul(grad_output, probabilities, summed, value=-1.0),
            "log_softmax backward fused multiply-subtract",
        )
        # addcmul has been enqueued on this device context. Drop its input
        # holders immediately so MAX's stream-ordered allocator can recycle
        # the vocabulary-sized probability temporary without a CPU/GPU sync.
        del probabilities, summed
        if grad_input._dtype != ctx.input_dtype:
            grad_input = aten_fast._cast_tensor(grad_input, ctx.input_dtype)
        return grad_input, None, None


@no_type_check
def _log_softmax_autograd(input, dim, half_to_float):
    if not input.requires_grad:
        return _require_handled(
            _fast().fast_aten__log_softmax(input, dim, half_to_float),
            "aten::_log_softmax",
        )
    return _LogSoftmaxAutograd.apply(input, dim, half_to_float)


class _LinearAutograd(torch.autograd.Function):
    @staticmethod
    @no_type_check
    def forward(ctx, input, weight, bias):
        output = _require_handled(
            _fast().fast_aten_linear(input, weight, bias), "aten::linear"
        )
        ctx.save_for_backward(input, weight)
        ctx.saved_payloads = (_SavedMojoPayload(input), _SavedMojoPayload(weight))
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    @no_type_check
    def backward(ctx, grad_output):
        aten_fast = _fast()
        input, weight = _restore_saved_mojo_tensors(ctx)
        input = input._contig()
        weight = weight._contig()
        input_features = input._shape[-1]
        output_features = weight._shape[0]
        rows = math.prod(input._shape[:-1]) if len(input._shape) > 1 else 1

        input2 = _contiguous_view(input, (rows, input_features))
        grad2 = _contiguous_view(grad_output, (rows, output_features))
        grad_input2 = _require_handled(
            aten_fast.fast_aten_mm(grad2, weight), "linear input gradient"
        )
        grad_input = _contiguous_view(grad_input2, input._shape)

        grad_t = _require_handled(
            aten_fast.fast_aten_transpose(grad2, 0, 1),
            "linear transpose output gradient",
        )
        grad_weight = _require_handled(
            aten_fast.fast_aten_mm(grad_t, input2), "linear weight gradient"
        )
        grad_bias = None
        if ctx.has_bias:
            grad_bias = _require_handled(
                aten_fast.fast_aten_sum(grad2, dim=[0], keepdim=False),
                "linear bias gradient",
            )
        return grad_input, grad_weight, grad_bias


@no_type_check
def _linear_autograd(input, weight, bias=None):
    if not (
        input.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    ):
        return _require_handled(
            _fast().fast_aten_linear(input, weight, bias), "aten::linear"
        )
    return _LinearAutograd.apply(input, weight, bias)


def register_autograd_ops() -> None:
    """Install the core AutogradPrivateUse1 kernels once."""
    global _registered
    if _registered:
        return

    torch.library.impl("aten::_log_softmax", "AutogradPrivateUse1")(
        _log_softmax_autograd
    )
    torch.library.impl("aten::linear", "AutogradPrivateUse1")(_linear_autograd)
    _registered = True
