"""Pure-Mojo fused dropout and softmax backward kernels for eager SDPA.

For every contiguous FP32 row, these kernels compute::

    masked_dP = dP_drop * Float32(mask) * Float32(dropout_scale)
    row_sum = sum(P * masked_dP)
    dScores = P * (masked_dP - row_sum) * Float32(score_scale)

The no-mask path omits the mask and dropout-scale operations completely.  A
block owns one row at a time and grid-strides over rows, so all shapes and
scalar values remain runtime-dynamic.  The implementation uses no scratch
allocation, host access, vendor library, or synchronization; launches are
enqueued asynchronously on the caller's ``DeviceContext``.
"""

from std.gpu import block_idx, grid_dim, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.primitives import block
from std.sys.info import has_accelerator

from op_utils import _enqueue_cached


comptime _BLOCK = 256
comptime _MAX_GRID = 65535


@__name("nanogpt_sdpa_dropout_softmax_backward_masked_f32")
def _masked_f32(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    probabilities: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_after_dropout: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mask: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    rows: Int,
    cols: Int,
    dropout_scale: Float32,
    score_scale: Float32,
):
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var row_stride = Int(grid_dim.x)
    while row < rows:
        var base = row * cols
        var partial_sum = Float32(0.0)
        var col = tid
        while col < cols:
            var index = base + col
            # Preserve the pinned IEEE operation order. In particular, a
            # false mask is multiplied rather than used as a selection, so
            # non-finite gradients still propagate as specified.
            var masked_d_p = (
                grad_after_dropout[index]
                * mask[index].cast[DType.float32]()
                * dropout_scale
            )
            partial_sum += probabilities[index] * masked_d_p
            col += _BLOCK

        # Every thread in the block reaches this row reduction. Broadcasting
        # keeps the result available for the independent output columns.
        var row_sum = block.sum[block_size=_BLOCK, broadcast=True](partial_sum)
        col = tid
        while col < cols:
            var index = base + col
            var masked_d_p = (
                grad_after_dropout[index]
                * mask[index].cast[DType.float32]()
                * dropout_scale
            )
            output[index] = (
                probabilities[index] * (masked_d_p - row_sum) * score_scale
            )
            col += _BLOCK
        row += row_stride


@__name("nanogpt_sdpa_dropout_softmax_backward_unmasked_f32")
def _unmasked_f32(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    probabilities: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_after_dropout: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    score_scale: Float32,
):
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var row_stride = Int(grid_dim.x)
    while row < rows:
        var base = row * cols
        var partial_sum = Float32(0.0)
        var col = tid
        while col < cols:
            var index = base + col
            partial_sum += probabilities[index] * grad_after_dropout[index]
            col += _BLOCK

        var row_sum = block.sum[block_size=_BLOCK, broadcast=True](partial_sum)
        col = tid
        while col < cols:
            var index = base + col
            output[index] = (
                probabilities[index]
                * (grad_after_dropout[index] - row_sum)
                * score_scale
            )
            col += _BLOCK
        row += row_stride


def enqueue_sdpa_dropout_softmax_backward_f32(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    probabilities: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_after_dropout: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mask: Optional[UnsafePointer[Scalar[DType.bool], MutAnyOrigin]],
    rows: Int,
    cols: Int,
    has_mask: Bool,
    dropout_scale: Float64,
    score_scale: Float64,
    ctx: DeviceContext,
) raises:
    # Check the nullable-mask ABI even for empty shapes, before any possible
    # launch. This catches contradictory host metadata deterministically.
    if has_mask:
        if not mask:
            raise Error(
                "sdpa_dropout_softmax_backward_kernels: has_mask requires"
                " a non-null mask"
            )
    elif mask:
        raise Error(
            "sdpa_dropout_softmax_backward_kernels: a mask requires"
            " has_mask=true"
        )

    comptime if has_accelerator():
        if rows <= 0 or cols <= 0:
            return

        var grid = min(rows, _MAX_GRID)
        var score_scale_f32 = Float32(score_scale)
        if has_mask:
            var mask_ptr = mask.value()
            _enqueue_cached[_masked_f32](
                ctx,
                "nanogpt_sdpa_dropout_softmax_backward_masked_f32",
                grid,
                1,
                1,
                _BLOCK,
                output,
                probabilities,
                grad_after_dropout,
                mask_ptr,
                rows,
                cols,
                Float32(dropout_scale),
                score_scale_f32,
            )
        else:
            _enqueue_cached[_unmasked_f32](
                ctx,
                "nanogpt_sdpa_dropout_softmax_backward_unmasked_f32",
                grid,
                1,
                1,
                _BLOCK,
                output,
                probabilities,
                grad_after_dropout,
                rows,
                cols,
                score_scale_f32,
            )
    else:
        raise Error("no GPU accelerator available at compile time")
