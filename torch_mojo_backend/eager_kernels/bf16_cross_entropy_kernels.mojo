from std.gpu import WARP_SIZE, block_dim, block_idx, grid_dim, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.primitives import warp
from std.gpu.primitives.warp import shuffle_down
from std.math import exp, log, log1p
from std.math.uutils import ufloordiv, umod
from std.memory import UnsafePointer
from std.utils.numerics import isnan, nan


comptime _MAX_MACHINE_INT = 9223372036854775807
comptime _THREADS = 256
comptime _WARPS_PER_BLOCK = _THREADS // WARP_SIZE
comptime _MAX_GRID_X = 65535


@__name("nanogpt_bf16_ce_forward_rows")
def nanogpt_bf16_ce_forward_rows(
    row_max: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    row_logsum: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    row_loss_scratch: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    logits: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    targets: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    rows: Int,
    classes: Int,
    ignore_index: Int,
):
    var tid = thread_idx.x
    var warp_in_block = ufloordiv(tid, WARP_SIZE)
    var lane = umod(tid, WARP_SIZE)
    var row = block_idx.x * _WARPS_PER_BLOCK + Int(warp_in_block)
    var row_stride = grid_dim.x * _WARPS_PER_BLOCK
    while row < rows:
        var row_base = row * classes
        var first_value = logits[row_base].cast[DType.float32]()
        var thread_maximum = first_value
        var thread_nan_count = Scalar[DType.float32](0.0)
        if isnan(first_value):
            thread_nan_count = 1.0
        var col = Int(lane)
        while col < classes:
            var value = logits[row_base + col].cast[DType.float32]()
            if isnan(value):
                thread_nan_count = 1.0
            elif not isnan(thread_maximum) and value > thread_maximum:
                thread_maximum = value
            col += WARP_SIZE

        var maximum = warp.max(thread_maximum)
        var nan_count = warp.sum(thread_nan_count)
        if nan_count > 0.0:
            maximum = nan[DType.float32]()

        var thread_exponential_sum = Scalar[DType.float32](0.0)
        col = Int(lane)
        while col < classes:
            var value = logits[row_base + col].cast[DType.float32]()
            thread_exponential_sum += exp(value - maximum)
            col += WARP_SIZE
        var exponential_sum = warp.sum(thread_exponential_sum)
        var logarithmic_sum = log(exponential_sum)
        if exponential_sum < 2.0:
            logarithmic_sum = log1p(exponential_sum - 1.0)

        if lane == 0:
            row_max[row] = maximum
            row_logsum[row] = logarithmic_sum

            var target_i64 = targets[row]
            var target = Int(target_i64)
            if target != ignore_index:
                debug_assert[assert_mode="safe"](
                    target >= 0 and target < classes,
                    "nanogpt_bf16_ce_target_out_of_bounds",
                )
                var target_logit = logits[row_base + target].cast[DType.float32]()
                var logp = Scalar[DType.bfloat16](
                    (target_logit - maximum) - logarithmic_sum
                )
                row_loss_scratch[row] = -logp.cast[DType.float32]()
            else:
                row_loss_scratch[row] = 0.0

        if rows - row <= row_stride:
            break
        row += row_stride


@__name("nanogpt_bf16_ce_forward_reduce")
def nanogpt_bf16_ce_forward_reduce(
    loss: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    total_weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    row_loss_scratch: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    targets: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    rows: Int,
    ignore_index: Int,
):
    var lane = Int(thread_idx.x)
    var loss_sum = Scalar[DType.float32](0.0)
    var weight_count = 0
    var row_base = 0
    while row_base < rows:
        var row = row_base + lane
        var thread_loss = Scalar[DType.float32](0.0)
        var thread_valid = Scalar[DType.float32](0.0)
        if row < rows:
            thread_loss = row_loss_scratch[row]
            if Int(targets[row]) != ignore_index:
                thread_valid = 1.0

        if lane == 0:
            loss_sum += thread_loss
            if thread_valid > 0.0:
                weight_count += 1

        var offset = 1
        while offset < WARP_SIZE:
            var other_loss = shuffle_down(thread_loss, UInt32(offset))
            var other_valid = shuffle_down(thread_valid, UInt32(offset))
            if lane == 0 and row_base + offset < rows:
                loss_sum += other_loss
                if other_valid > 0.0:
                    weight_count += 1
            offset += 1

        if rows - row_base <= WARP_SIZE:
            break
        row_base += WARP_SIZE

    if lane == 0:
        var weight_sum = Scalar[DType.float32](weight_count)
        total_weight[0] = weight_sum
        loss[0] = loss_sum / weight_sum


@__name("nanogpt_bf16_ce_backward_rows")
def nanogpt_bf16_ce_backward_rows(
    grad_input: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    logits: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    targets: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    row_max: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    row_logsum: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    total_weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    classes: Int,
    ignore_index: Int,
):
    var row = block_idx.x
    var row_stride = grid_dim.x
    while row < rows:
        var target = Int(targets[row])
        var ignored = target == ignore_index
        if not ignored:
            debug_assert[assert_mode="safe"](
                target >= 0 and target < classes,
                "nanogpt_bf16_ce_target_out_of_bounds",
            )

        var maximum = row_max[row]
        var logarithmic_sum = row_logsum[row]
        var scale = Scalar[DType.bfloat16](0.0)
        if not ignored:
            scale = Scalar[DType.bfloat16](-grad_output[0] / total_weight[0])
        var scale_f32 = scale.cast[DType.float32]()
        var row_base = row * classes
        var col = thread_idx.x
        while col < classes:
            var value = logits[row_base + col].cast[DType.float32]()
            var logp = Scalar[DType.bfloat16](
                (value - maximum) - logarithmic_sum
            )
            var probability = exp(logp.cast[DType.float32]())
            var nll_gradient = Scalar[DType.float32](0.0)
            if not ignored and col == target:
                nll_gradient = scale_f32
            grad_input[row_base + col] = Scalar[DType.bfloat16](
                nll_gradient - probability * scale_f32
            )
            if classes - col <= block_dim.x:
                break
            col += block_dim.x

        if rows - row <= row_stride:
            break
        row += row_stride


def _check_spans(rows: Int, classes: Int) raises:
    if rows <= 0:
        raise Error("rows must be positive")
    if classes <= 0:
        raise Error("classes must be positive")
    if rows > _MAX_MACHINE_INT // 8:
        raise Error("row tensor span exceeds machine Int")
    if rows > (_MAX_MACHINE_INT // 2) // classes:
        raise Error("logit tensor span exceeds machine Int")


def enqueue_bf16_cross_entropy_forward(
    loss: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    total_weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    row_max: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    row_logsum: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    row_loss_scratch: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    logits: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    targets: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    rows: Int,
    classes: Int,
    ignore_index: Int,
    ctx: DeviceContext,
) raises:
    _check_spans(rows, classes)
    var blocks = rows // _WARPS_PER_BLOCK
    if rows % _WARPS_PER_BLOCK != 0:
        blocks += 1
    if blocks > _MAX_GRID_X:
        blocks = _MAX_GRID_X
    ctx.enqueue_function[nanogpt_bf16_ce_forward_rows](
        row_max,
        row_logsum,
        row_loss_scratch,
        logits,
        targets,
        rows,
        classes,
        ignore_index,
        grid_dim=blocks,
        block_dim=_THREADS,
    )
    ctx.enqueue_function[nanogpt_bf16_ce_forward_reduce](
        loss,
        total_weight,
        row_loss_scratch,
        targets,
        rows,
        ignore_index,
        grid_dim=1,
        block_dim=WARP_SIZE,
    )


def enqueue_bf16_cross_entropy_backward(
    grad_input: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    logits: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    targets: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    row_max: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    row_logsum: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    total_weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    classes: Int,
    ignore_index: Int,
    ctx: DeviceContext,
) raises:
    _check_spans(rows, classes)
    var blocks = rows
    if blocks > _MAX_GRID_X:
        blocks = _MAX_GRID_X
    ctx.enqueue_function[nanogpt_bf16_ce_backward_rows](
        grad_input,
        grad_output,
        logits,
        targets,
        row_max,
        row_logsum,
        total_weight,
        rows,
        classes,
        ignore_index,
        grid_dim=blocks,
        block_dim=_THREADS,
    )
