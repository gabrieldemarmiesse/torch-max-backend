"""AutogradPrivateUse1 plumbing for eager Mojo operations.

``TorchMojoTensor`` is a true wrapper subclass, so PyTorch's generated autograd
can be used whenever it exposes a suitable native backward ATen operation. The
custom Functions left here cover fused formulas or unsupported native routes;
their sidecar metadata also validates saved-tensor hook results and restores
values that hooks deliberately unpack onto the host.
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


# Native EmbeddingBackward0 already reaches our dense backward kernel for the
# supported first-order case, but it can enter unsupported sparse/frequency
# paths without a safe backend error, and its double backward needs index_select.
# Keep this preflight boundary until all three native routes are safe.
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


# cross_entropy_loss has no native backward ATen op through which PyTorch could
# pass the fused forward's private row statistics, so this custom node is the
# actual fused-operation boundary rather than redundant dispatcher plumbing.
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


# Eligible FA4 calls are routed through PyTorch's native lower flash pair below.
# This custom node remains only for the generic math/dropout implementation,
# whose fused intermediate-saving backward has no native ATen schema.
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
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    @no_type_check
    def backward(ctx, grad_output):
        aten_fast = _fast()
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
    needs_backward = torch.is_grad_enabled() and (
        query.requires_grad or key.requires_grad or value.requires_grad
    )
    aten_fast = _fast()
    # The eligible FA4 regime already implements PyTorch's lower flash forward
    # and backward pair. Dispatch through it so generated autograd owns the
    # saves, version checks, and backward call; retain the custom Function only
    # for the generic math/dropout fallback, which has no native backward op.
    if (
        needs_backward
        and aten_fast._fa4_bf16_d64_causal_inputs(
            query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
        )
        is not None
    ):
        return torch.ops.aten._scaled_dot_product_flash_attention.default(
            query, key, value, dropout_p, is_causal, False, scale=scale
        )[0]
    if not needs_backward:
        return _require_handled(
            aten_fast.fast_aten_scaled_dot_product_attention(
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

    torch.library.impl("aten::cross_entropy_loss", "AutogradPrivateUse1")(
        _cross_entropy_loss_autograd
    )
    torch.library.impl("aten::embedding", "AutogradPrivateUse1")(_embedding_autograd)
    torch.library.impl("aten::scaled_dot_product_attention", "AutogradPrivateUse1")(
        _scaled_dot_product_attention_autograd
    )
    _registered = True
