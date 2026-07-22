"""AutogradPrivateUse1 plumbing for eager Mojo operations.

``TorchMojoTensor`` is a true wrapper subclass, so PyTorch's saved-variable
plumbing now preserves its Python allocation payload. These custom Functions
still provide backward formulas that are not registered as native Mojo ATen
operators. Their sidecar metadata also validates saved-tensor hook results and
restores values that hooks deliberately unpack onto the host.
"""

import math
from typing import no_type_check

import torch
from torch.autograd.function import once_differentiable

from .cross_entropy import decompose_cross_entropy_loss
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
    """Metadata used to validate or restore a PyTorch-unpacked saved variable.

    The ordinary wrapper-subclass path returns a fresh, complete
    ``TorchMojoTensor`` with the shared version counter and allocation payload.
    The recorded fields validate that object without retaining a Tensor
    directly on ``ctx``.

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

        # Compatibility path for a PyTorch version that unpacks a plain
        # PrivateUse1 tensor despite the wrapper subclass. SavedVariable has
        # already checked the original shared version counter at this point;
        # construct a valid wrapper instead of attempting an invalid class swap.
        return TorchMojoTensor._make(
            self.holder,
            self.ptr,
            self.shape,
            self.strides,
            self.offset,
            self.dtype,
            self.device,
            contiguous=self.is_contiguous,
        )

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


class _EmbeddingAutograd(torch.autograd.Function):
    @staticmethod
    @no_type_check
    def forward(ctx, weight, indices, padding_idx, scale_grad_by_freq, sparse):
        if sparse:
            raise NotImplementedError(
                "Mojo eager embedding autograd does not yet support sparse=True"
            )
        if scale_grad_by_freq:
            raise NotImplementedError(
                "Mojo eager embedding autograd does not yet support "
                "scale_grad_by_freq=True"
            )
        output = _require_handled(
            _fast().fast_aten_embedding(
                weight, indices, padding_idx, scale_grad_by_freq, sparse
            ),
            "aten::embedding",
        )
        # Weight values are irrelevant to its gradient. Saving only indices
        # preserves their version check and avoids retaining the table twice.
        ctx.save_for_backward(indices)
        ctx.saved_payloads = (_SavedMojoPayload(indices),)
        ctx.num_weights = weight._shape[0]
        ctx.padding_idx = padding_idx
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    @no_type_check
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None, None
        (indices,) = _restore_saved_mojo_tensors(ctx)
        grad_weight = _require_handled(
            _fast().fast_aten_embedding_dense_backward(
                grad_output, indices, ctx.num_weights, ctx.padding_idx, False
            ),
            "aten::embedding_dense_backward",
        )
        return grad_weight, None, None, None, None


@no_type_check
def _embedding_autograd(
    weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False
):
    if not weight.requires_grad:
        return _require_handled(
            _fast().fast_aten_embedding(
                weight, indices, padding_idx, scale_grad_by_freq, sparse
            ),
            "aten::embedding",
        )
    return _EmbeddingAutograd.apply(
        weight, indices, padding_idx, scale_grad_by_freq, sparse
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


class _NativeDropoutAutograd(torch.autograd.Function):
    @staticmethod
    @no_type_check
    def forward(ctx, input, p, train):
        output, mask = _require_handled(
            _fast().fast_aten_native_dropout(input, p, train), "aten::native_dropout"
        )
        # Native dropout's input derivative depends only on the returned mask.
        # Saving it normally preserves PyTorch's version-counter checks; the
        # payload restores the eager Mojo allocation after SavedVariable unpack.
        ctx.save_for_backward(mask)
        ctx.saved_payloads = (_SavedMojoPayload(mask),)
        ctx.scale = (0.0 if p == 1.0 else 1.0 / (1.0 - p)) if train is True else 1.0
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(mask)
        return output, mask

    @staticmethod
    @no_type_check
    def backward(ctx, grad_output, _grad_mask):
        (mask,) = _restore_saved_mojo_tensors(ctx)
        grad_input = _require_handled(
            _fast().fast_aten_native_dropout_backward(grad_output, mask, ctx.scale),
            "aten::native_dropout_backward",
        )
        return grad_input, None, None


@no_type_check
def _native_dropout_autograd(input, p, train):
    if not input.requires_grad:
        return _require_handled(
            _fast().fast_aten_native_dropout(input, p, train), "aten::native_dropout"
        )
    return _NativeDropoutAutograd.apply(input, p, train)


class _NativeLayerNormAutograd(torch.autograd.Function):
    @staticmethod
    @no_type_check
    def forward(ctx, input, normalized_shape, weight, bias, eps):
        output, mean, rstd = _require_handled(
            _fast().fast_aten_native_layer_norm(
                input, normalized_shape, weight, bias, eps
            ),
            "aten::native_layer_norm",
        )
        saved = [input, mean, rstd]
        if weight is not None:
            saved.append(weight)
        if bias is not None:
            saved.append(bias)
        ctx.save_for_backward(*saved)
        ctx.saved_payloads = tuple(_SavedMojoPayload(tensor) for tensor in saved)
        ctx.normalized_shape = tuple(normalized_shape)
        ctx.has_weight = weight is not None
        ctx.has_bias = bias is not None
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(mean, rstd)
        return output, mean, rstd

    @staticmethod
    @no_type_check
    def backward(ctx, grad_output, _grad_mean, _grad_rstd):
        # Restore every saved tensor first. Accessing ctx.saved_tensors performs
        # the version checks even when a particular value is not needed by the
        # requested gradient mask (matching native LayerNorm's safety rules).
        saved = iter(_restore_saved_mojo_tensors(ctx))
        input = next(saved)
        mean = next(saved)
        rstd = next(saved)
        weight = next(saved) if ctx.has_weight else None
        bias = next(saved) if ctx.has_bias else None

        output_mask = (
            bool(ctx.needs_input_grad[0]),
            bool(ctx.needs_input_grad[2]) and ctx.has_weight,
            bool(ctx.needs_input_grad[3]) and ctx.has_bias,
        )
        grad_input, grad_weight, grad_bias = _require_handled(
            _fast().fast_aten_native_layer_norm_backward(
                grad_output,
                input,
                ctx.normalized_shape,
                mean,
                rstd,
                weight,
                bias,
                output_mask,
            ),
            "aten::native_layer_norm_backward",
        )
        return grad_input, None, grad_weight, grad_bias, None


@no_type_check
def _native_layer_norm_autograd(input, normalized_shape, weight, bias, eps):
    if not (
        input.requires_grad
        or (weight is not None and weight.requires_grad)
        or (bias is not None and bias.requires_grad)
    ):
        return _require_handled(
            _fast().fast_aten_native_layer_norm(
                input, normalized_shape, weight, bias, eps
            ),
            "aten::native_layer_norm",
        )
    return _NativeLayerNormAutograd.apply(
        input, tuple(normalized_shape), weight, bias, eps
    )


class _Bf16CrossEntropyAutograd(torch.autograd.Function):
    @staticmethod
    @no_type_check
    def forward(ctx, input, target, weight, reduction, ignore_index, label_smoothing):
        aten_fast = _fast()
        result = aten_fast.fast_bf16_cross_entropy_forward(
            input,
            target,
            weight,
            reduction,
            ignore_index,
            label_smoothing,
            require_backward=True,
        )
        if result is None or result is aten_fast.NOT_HANDLED:
            raise RuntimeError(
                "fused BF16 cross entropy became unavailable after host preflight"
            )
        loss, row_max, row_logsum, total_weight = result
        ctx.save_for_backward(input, target, row_max, row_logsum, total_weight)
        ctx.saved_payloads = tuple(
            _SavedMojoPayload(tensor)
            for tensor in (input, target, row_max, row_logsum, total_weight)
        )
        ctx.ignore_index = ignore_index
        ctx.set_materialize_grads(False)
        return loss

    @staticmethod
    @once_differentiable
    @no_type_check
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None, None, None
        input, target, row_max, row_logsum, total_weight = _restore_saved_mojo_tensors(
            ctx
        )
        grad_input = _require_handled(
            _fast().fast_bf16_cross_entropy_backward(
                grad_output,
                input,
                target,
                row_max,
                row_logsum,
                total_weight,
                ctx.ignore_index,
            ),
            "aten::cross_entropy_loss backward",
        )
        return grad_input, None, None, None, None, None


@no_type_check
def _cross_entropy_loss_autograd(
    input, target, weight=None, reduction=1, ignore_index=-100, label_smoothing=0.0
):
    """Route the narrow fused path or invoke PyTorch's composite directly."""
    aten_fast = _fast()
    needs_backward = torch.is_grad_enabled() and bool(input.requires_grad)
    if not aten_fast.bf16_cross_entropy_supported(
        input,
        target,
        weight,
        reduction,
        ignore_index,
        label_smoothing,
        require_backward=needs_backward,
    ):
        # This intentionally preserves PyTorch's exact composite dispatch and
        # dtype semantics.  The backend's pre-existing BF16 NLL coverage
        # outside autocast is a separate limitation; do not silently replace
        # that path with this fused bridge's FP32-loss contract.
        return decompose_cross_entropy_loss(
            input, target, weight, reduction, ignore_index, label_smoothing
        )
    if needs_backward:
        return _Bf16CrossEntropyAutograd.apply(
            input, target, weight, reduction, ignore_index, label_smoothing
        )
    result = aten_fast.fast_bf16_cross_entropy_forward(
        input, target, weight, reduction, ignore_index, label_smoothing
    )
    if result is None or result is aten_fast.NOT_HANDLED:
        raise RuntimeError(
            "fused BF16 cross entropy became unavailable after host preflight"
        )
    loss, _row_max, _row_logsum, _total_weight = result
    return loss


@no_type_check
def _nll_loss_forward_impl(input, target, weight, reduction, ignore_index):
    """Allocate the two functional NLL outputs and enqueue the direct kernel."""
    aten_fast = _fast()
    inputs = aten_fast._nll_loss_inputs(input, target, weight, reduction, ignore_index)
    if inputs is None:
        raise NotImplementedError(
            "aten::nll_loss_forward is not supported by Mojo eager autograd "
            "for these inputs"
        )

    log_probs, _, rows, _ = inputs
    output_shape = (rows,) if reduction == 0 else ()
    output = TorchMojoTensor._alloc(output_shape, log_probs._dtype, log_probs._device)
    total_weight = TorchMojoTensor._alloc((), log_probs._dtype, log_probs._device)
    result = aten_fast.fast_aten_nll_loss_forward_output(
        log_probs,
        target,
        weight,
        reduction,
        ignore_index,
        output=output,
        total_weight=total_weight,
    )
    if result is None or result is aten_fast.NOT_HANDLED:
        raise NotImplementedError(
            "aten::nll_loss_forward is not supported by Mojo eager autograd "
            "for these inputs"
        )
    return result


class _NllLossForwardAutograd(torch.autograd.Function):
    @staticmethod
    @no_type_check
    def forward(ctx, input, target, weight, reduction, ignore_index):
        output, total_weight = _nll_loss_forward_impl(
            input, target, weight, reduction, ignore_index
        )
        ctx.save_for_backward(input, target, total_weight)
        ctx.saved_payloads = tuple(
            _SavedMojoPayload(tensor) for tensor in (input, target, total_weight)
        )
        ctx.reduction = reduction
        ctx.ignore_index = ignore_index
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(total_weight)
        return output, total_weight

    @staticmethod
    @no_type_check
    def backward(ctx, grad_output, _grad_total_weight):
        aten_fast = _fast()
        input, target, total_weight = _restore_saved_mojo_tensors(ctx)
        grad_input = TorchMojoTensor._alloc(input._shape, input._dtype, input._device)
        result = aten_fast.fast_aten_nll_loss_backward_grad_input(
            grad_output,
            input,
            target,
            None,
            ctx.reduction,
            ctx.ignore_index,
            total_weight,
            grad_input=grad_input,
        )
        _require_handled(result, "aten::nll_loss_backward")
        return grad_input, None, None, None, None


@no_type_check
def _nll_loss_forward_autograd(input, target, weight, reduction, ignore_index):
    if weight is not None:
        # The direct eager kernel currently has an explicit no-weight contract.
        raise NotImplementedError(
            "aten::nll_loss_forward is not supported by Mojo eager autograd "
            "for weighted inputs"
        )
    if not input.requires_grad:
        return _nll_loss_forward_impl(input, target, weight, reduction, ignore_index)
    return _NllLossForwardAutograd.apply(input, target, weight, reduction, ignore_index)


class _ScaledDotProductAttentionAutograd(torch.autograd.Function):
    @staticmethod
    @no_type_check
    def forward(
        ctx, query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
    ):
        if attn_mask is not None or enable_gqa:
            raise NotImplementedError(
                "Mojo eager SDPA autograd currently supports no attention mask, "
                "and enable_gqa=False"
            )

        aten_fast = _fast()
        fa4_result = aten_fast.fast_fa4_bf16_d64_causal_forward(
            query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
        )
        if fa4_result is not aten_fast.NOT_HANDLED:
            output, logsumexp, q_native, k_native, v_native = fa4_result
            # Physical BTHD wrappers only need to survive the enqueued
            # forward. Retaining copies across every transformer layer would
            # duplicate unsupported QKV layouts; re-prepare from the
            # version-checked public views immediately before backward.
            del q_native, k_native, v_native
            saved = (query, key, value, output, logsumexp)
            ctx.save_for_backward(*saved)
            ctx.saved_payloads = tuple(_SavedMojoPayload(tensor) for tensor in saved)
            ctx.saved_names = ("query", "key", "value", "output", "logsumexp")
            ctx.needed_input_gradients = tuple(
                bool(ctx.needs_input_grad[index]) for index in range(3)
            )
            ctx.scale = (
                float(scale) if scale is not None else 1.0 / math.sqrt(query._shape[-1])
            )
            ctx.fa4 = True
            ctx.set_materialize_grads(False)
            return output

        result = aten_fast._sdpa_math_forward_with_dropout(
            query, key, value, is_causal, scale, dropout_p
        )
        if result is aten_fast.NOT_HANDLED:
            raise NotImplementedError(
                "aten::scaled_dot_product_attention is not supported by Mojo "
                "eager autograd for these inputs"
            )
        output, probabilities, dropout_mask = result
        need_query, need_key, need_value = (
            bool(ctx.needs_input_grad[index]) for index in range(3)
        )
        saved = []
        saved_names = []

        def save(name, tensor):
            saved_names.append(name)
            saved.append(tensor)

        # dQ consumes K and V; dK consumes Q and V; dV consumes only P_drop.
        # Every requested input gradient needs the pre-dropout probabilities.
        if need_key:
            save("query", query)
        if need_query:
            save("key", key)
        if need_query or need_key:
            save("value", value)
        save("probabilities", probabilities)
        if dropout_mask is not None:
            save("dropout_mask", dropout_mask)
        ctx.save_for_backward(*saved)
        ctx.saved_payloads = tuple(_SavedMojoPayload(tensor) for tensor in saved)
        ctx.saved_names = tuple(saved_names)
        ctx.needed_input_gradients = (need_query, need_key, need_value)
        ctx.query_shape = tuple(query._shape)
        ctx.key_shape = tuple(key._shape)
        ctx.value_shape = tuple(value._shape)
        ctx.scale = (
            float(scale) if scale is not None else 1.0 / math.sqrt(query._shape[-1])
        )
        ctx.has_dropout = dropout_mask is not None
        ctx.dropout_scale = (
            (0.0 if float(dropout_p) == 1.0 else 1.0 / (1.0 - float(dropout_p)))
            if ctx.has_dropout
            else 1.0
        )
        ctx.fa4 = False
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    @no_type_check
    def backward(ctx, grad_output):
        aten_fast = _fast()
        if getattr(ctx, "fa4", False):
            if grad_output is None:
                return (None,) * 8
            restored = _restore_saved_mojo_tensors(ctx)
            saved = dict(zip(ctx.saved_names, restored, strict=True))
            # Unpacking invokes PyTorch's version checks. Eligible fused QKV
            # become zero-copy BTHD views here; unsupported layouts retain the
            # old contiguous materialization path.
            query = saved.pop("query")
            key = saved.pop("key")
            value = saved.pop("value")
            q_native = aten_fast._fa4_native_bthd(query)
            k_native = aten_fast._fa4_native_bthd(key)
            v_native = aten_fast._fa4_native_bthd(value)
            if q_native is None or k_native is None or v_native is None:
                raise RuntimeError("FA4 saved inputs could not be prepared as BTHD")
            gradients = _require_handled(
                aten_fast.fast_fa4_bf16_d64_causal_backward(
                    q_native,
                    k_native,
                    v_native,
                    saved.pop("output"),
                    saved.pop("logsumexp"),
                    grad_output,
                    ctx.scale,
                ),
                "FA4 scaled_dot_product_attention backward",
            )
            if saved:
                raise RuntimeError(f"unused Mojo FA4 saved tensors: {tuple(saved)}")
            need_query, need_key, need_value = ctx.needed_input_gradients
            grad_query, grad_key, grad_value = gradients
            return (
                grad_query if need_query else None,
                grad_key if need_key else None,
                grad_value if need_value else None,
                None,
                None,
                None,
                None,
                None,
            )

        restored = _restore_saved_mojo_tensors(ctx)
        saved = dict(zip(ctx.saved_names, restored, strict=True))
        del restored
        need_query, need_key, need_value = ctx.needed_input_gradients
        probabilities = saved.pop("probabilities")
        dropout_mask = saved.pop("dropout_mask", None)
        batch, heads, query_length, head_dim = ctx.query_shape
        key_length = ctx.key_shape[2]
        batch_heads = batch * heads

        p3 = _contiguous_view(probabilities, (batch_heads, query_length, key_length))
        grad3 = _contiguous_view(grad_output, (batch_heads, query_length, head_dim))
        del probabilities

        mask3 = None
        if ctx.has_dropout:
            mask3 = _contiguous_view(
                dropout_mask, (batch_heads, query_length, key_length)
            )
            del dropout_mask

        grad_query3 = None
        grad_key3 = None
        grad_value3 = None

        if need_value:
            effective_p3 = p3
            if ctx.has_dropout:
                effective_p3 = _require_handled(
                    aten_fast.fast_aten_native_dropout_backward(
                        p3, mask3, ctx.dropout_scale
                    ),
                    "SDPA reconstruct dropped probabilities",
                )

            # dV = P_drop^T @ grad = (grad^T @ P_drop)^T. Materializing
            # grad^T costs O(L*E), rather than materializing P_drop^T at
            # O(L*S). The final transpose copy is the required contiguous dV.
            grad_t = _require_handled(
                aten_fast.fast_aten_transpose(grad3, 1, 2),
                "SDPA transpose output gradient",
            )
            grad_t = _contiguous_view(grad_t, (batch_heads, head_dim, query_length))
            grad_value_t = _require_handled(
                aten_fast.fast_aten_bmm(grad_t, effective_p3), "SDPA value gradient"
            )
            grad_value_view = _require_handled(
                aten_fast.fast_aten_transpose(grad_value_t, 1, 2),
                "SDPA transpose value gradient",
            )
            grad_value3 = _contiguous_view(
                grad_value_view, (batch_heads, key_length, head_dim)
            )
            del grad_t, grad_value_t, grad_value_view
            if effective_p3 is not p3:
                del effective_p3

        if need_query or need_key:
            v = saved.pop("value")._contig()
            v3 = _contiguous_view(v, (batch_heads, key_length, head_dim))
            # dP_drop = grad @ V^T. BmmSpec's logical transpose flag keeps V
            # dense and avoids materializing its (E,S) transpose.
            grad_effective_probabilities = _require_handled(
                aten_fast._fast_aten_bmm_transpose_b(grad3, v3),
                "SDPA probability gradient",
            )
            del v, v3
            grad_scores = aten_fast.fast_sdpa_dropout_softmax_backward(
                p3,
                grad_effective_probabilities,
                mask3 if ctx.has_dropout else None,
                ctx.dropout_scale,
                ctx.scale,
            )
            if grad_scores is aten_fast.NOT_HANDLED:
                # Keep the pre-fusion composition as the exact compatibility
                # route for non-FP32, non-GPU, or otherwise unsupported eager
                # inputs.  Dropout stays before P multiplication and the row
                # reduction, matching the fused arithmetic contract.
                effective_p3 = (
                    _require_handled(
                        aten_fast.fast_aten_native_dropout_backward(
                            grad_effective_probabilities, mask3, ctx.dropout_scale
                        ),
                        "SDPA dropout gradient",
                    )
                    if ctx.has_dropout
                    else grad_effective_probabilities
                )
                if effective_p3 is not grad_effective_probabilities:
                    del grad_effective_probabilities

                weighted = _require_handled(
                    aten_fast.fast_aten_mul(effective_p3, p3),
                    "SDPA softmax weighted gradient",
                )
                row_sum = _require_handled(
                    aten_fast.fast_aten_sum(weighted, dim=[2], keepdim=True),
                    "SDPA softmax gradient reduction",
                )
                del weighted
                centered = _require_handled(
                    aten_fast.fast_aten_sub(effective_p3, row_sum),
                    "SDPA softmax centered gradient",
                )
                del effective_p3, row_sum
                unscaled_grad_scores = _require_handled(
                    aten_fast.fast_aten_mul(centered, p3), "SDPA softmax gradient"
                )
                del centered
                grad_scores = _require_handled(
                    aten_fast.fast_aten_mul(unscaled_grad_scores, ctx.scale),
                    "SDPA scale gradient",
                )
                del unscaled_grad_scores
            else:
                del grad_effective_probabilities

            if need_query:
                k = saved.pop("key")._contig()
                k3 = _contiguous_view(k, (batch_heads, key_length, head_dim))
                grad_query3 = _require_handled(
                    aten_fast.fast_aten_bmm(grad_scores, k3), "SDPA query gradient"
                )
                del k, k3

            if need_key:
                q = saved.pop("query")._contig()
                q3 = _contiguous_view(q, (batch_heads, query_length, head_dim))
                # dK = dScores^T @ Q = (Q^T @ dScores)^T. Only Q^T's
                # O(L*E) storage is materialized; the final copy makes dK
                # contiguous in its required (S,E) layout.
                q_t = _require_handled(
                    aten_fast.fast_aten_transpose(q3, 1, 2), "SDPA transpose query"
                )
                q_t = _contiguous_view(q_t, (batch_heads, head_dim, query_length))
                grad_key_t = _require_handled(
                    aten_fast.fast_aten_bmm(q_t, grad_scores), "SDPA key gradient"
                )
                grad_key_view = _require_handled(
                    aten_fast.fast_aten_transpose(grad_key_t, 1, 2),
                    "SDPA transpose key gradient",
                )
                grad_key3 = _contiguous_view(
                    grad_key_view, (batch_heads, key_length, head_dim)
                )
                del q, q3, q_t, grad_key_t, grad_key_view
            del grad_scores

        del grad3, p3, mask3
        if saved:
            raise RuntimeError(f"unused Mojo SDPA saved tensors: {tuple(saved)}")

        grad_query = (
            _contiguous_view(grad_query3, ctx.query_shape) if need_query else None
        )
        grad_key = _contiguous_view(grad_key3, ctx.key_shape) if need_key else None
        grad_value = (
            _contiguous_view(grad_value3, ctx.value_shape) if need_value else None
        )
        return grad_query, grad_key, grad_value, None, None, None, None, None


@no_type_check
def _scaled_dot_product_attention_autograd(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    enable_gqa=False,
):
    if not (query.requires_grad or key.requires_grad or value.requires_grad):
        return _require_handled(
            _fast().fast_aten_scaled_dot_product_attention(
                query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
            ),
            "aten::scaled_dot_product_attention",
        )
    return _ScaledDotProductAttentionAutograd.apply(
        query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
    )


def register_autograd_ops() -> None:
    """Install concrete AutogradPrivateUse1 kernels once."""
    global _registered
    if _registered:
        return

    torch.library.impl("aten::_log_softmax", "AutogradPrivateUse1")(
        _log_softmax_autograd
    )
    torch.library.impl("aten::cross_entropy_loss", "AutogradPrivateUse1")(
        _cross_entropy_loss_autograd
    )
    torch.library.impl("aten::embedding", "AutogradPrivateUse1")(_embedding_autograd)
    torch.library.impl("aten::linear", "AutogradPrivateUse1")(_linear_autograd)
    torch.library.impl("aten::native_dropout", "AutogradPrivateUse1")(
        _native_dropout_autograd
    )
    torch.library.impl("aten::native_layer_norm", "AutogradPrivateUse1")(
        _native_layer_norm_autograd
    )
    torch.library.impl("aten::nll_loss_forward", "AutogradPrivateUse1")(
        _nll_loss_forward_autograd
    )
    torch.library.impl("aten::scaled_dot_product_attention", "AutogradPrivateUse1")(
        _scaled_dot_product_attention_autograd
    )
    _registered = True
