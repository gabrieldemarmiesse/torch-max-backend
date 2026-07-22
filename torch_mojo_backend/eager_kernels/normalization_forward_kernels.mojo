"""Correct baseline for the frozen native LayerNorm-forward harness.

This deliberately mirrors the production kernel's block-per-row structure so
the optimization agent starts from a working, runtime-dynamic implementation.
The frozen contract and harness, not this editable file, define acceptance.
"""

from std.ffi import _get_global_or_null, external_call
from std.gpu import block_idx, grid_dim, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.primitives import block
from std.math import min, sqrt
from std.memory import alloc


comptime _BLOCK = 256
comptime _VEC = 4
comptime _VEC_BLOCK = 128
comptime _MAX_GRID = 65535


@__name("layer_norm_forward_f32_vec4")
def _layer_norm_forward_f32_vec4[
    has_weight: Bool, has_bias: Bool
](
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    epsilon: Float32,
):
    """Retain up to two aligned vectors per lane across both reductions."""
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var row_stride = Int(grid_dim.x)
    while row < rows:
        var base = row * cols
        var col0 = tid * _VEC
        var col1 = (tid + _VEC_BLOCK) * _VEC
        var values0 = SIMD[DType.float32, _VEC](0.0)
        var values1 = SIMD[DType.float32, _VEC](0.0)
        if col0 < cols:
            values0 = input.load[width=_VEC, alignment=16](base + col0)
        if col1 < cols:
            values1 = input.load[width=_VEC, alignment=16](base + col1)

        var row_mean = block.sum[block_size=_VEC_BLOCK, broadcast=True](
            values0.reduce_add() + values1.reduce_add()
        ) / Float32(cols)
        var centered0 = values0 - row_mean
        var centered1 = values1 - row_mean
        var thread_variance = Float32(0.0)
        if col0 < cols:
            thread_variance += (centered0 * centered0).reduce_add()
        if col1 < cols:
            thread_variance += (centered1 * centered1).reduce_add()
        var variance = block.sum[block_size=_VEC_BLOCK, broadcast=True](
            thread_variance
        ) / Float32(cols)
        var row_rstd = 1.0 / sqrt(variance + epsilon)
        # A centered-square reduction is nonnegative for every finite row, so
        # no clamp is needed.  Preserve a non-finite reduction as NaN and use
        # it for both statistics: ATen reports NaN mean/rstd for rows that
        # contain NaN or infinity (including the all-infinity case).
        if variance != variance:
            row_mean = variance

        if tid == 0:
            mean[row] = row_mean
            rstd[row] = row_rstd
        if col0 < cols:
            var result0 = centered0 * row_rstd
            comptime if has_weight:
                result0 *= weight.load[width=_VEC, alignment=16](col0)
            comptime if has_bias:
                result0 += bias.load[width=_VEC, alignment=16](col0)
            output.store[width=_VEC, alignment=16](base + col0, result0)
        if col1 < cols:
            var result1 = centered1 * row_rstd
            comptime if has_weight:
                result1 *= weight.load[width=_VEC, alignment=16](col1)
            comptime if has_bias:
                result1 += bias.load[width=_VEC, alignment=16](col1)
            output.store[width=_VEC, alignment=16](base + col1, result1)
        row += row_stride


@__name("layer_norm_forward_f32_baseline")
def _layer_norm_forward_f32_baseline(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    epsilon: Float32,
    has_weight: Int,
    has_bias: Int,
):
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var row_stride = Int(grid_dim.x)
    while row < rows:
        var base = row * cols
        var thread_sum = Float32(0.0)
        var col = tid
        while col < cols:
            thread_sum += input[base + col]
            col += _BLOCK
        var row_mean = block.sum[block_size=_BLOCK, broadcast=True](
            thread_sum
        ) / Float32(cols)

        var thread_variance = Float32(0.0)
        col = tid
        while col < cols:
            var centered = input[base + col] - row_mean
            thread_variance += centered * centered
            col += _BLOCK
        var variance = block.sum[block_size=_BLOCK, broadcast=True](
            thread_variance
        ) / Float32(cols)
        var row_rstd = 1.0 / sqrt(variance + epsilon)
        if variance != variance:
            row_mean = variance
        if tid == 0:
            mean[row] = row_mean
            rstd[row] = row_rstd

        col = tid
        while col < cols:
            var value = (input[base + col] - row_mean) * row_rstd
            if has_weight != 0:
                value *= weight[col]
            if has_bias != 0:
                value += bias[col]
            output[base + col] = value
            col += _BLOCK
        row += row_stride


def _enqueue_vec4_cached[
    has_weight: Bool, has_bias: Bool
](
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    epsilon: Float32,
    ctx: DeviceContext,
) raises:
    var cache_name = String(
        t"LAYER_NORM_FORWARD_F32_VEC4_V1_{has_weight}_{has_bias}_{ctx.id()}"
    )
    comptime FuncT = type_of(
        ctx.compile_function[
            _layer_norm_forward_f32_vec4[has_weight, has_bias]
        ]()
    )
    if global_ptr := _get_global_or_null(cache_name):
        var cached = global_ptr.value().bitcast[FuncT]()
        ctx.enqueue_function(
            cached[],
            output,
            mean,
            rstd,
            input,
            weight,
            bias,
            rows,
            cols,
            epsilon,
            grid_dim=(min(rows, _MAX_GRID),),
            block_dim=(_VEC_BLOCK,),
        )
        return
    var compiled = ctx.compile_function[
        _layer_norm_forward_f32_vec4[has_weight, has_bias]
    ]()
    var cached = alloc[FuncT](1)
    cached.init_pointee_move(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_name), cached.bitcast[NoneType]()
    )
    ctx.enqueue_function(
        cached[],
        output,
        mean,
        rstd,
        input,
        weight,
        bias,
        rows,
        cols,
        epsilon,
        grid_dim=(min(rows, _MAX_GRID),),
        block_dim=(_VEC_BLOCK,),
    )


def _enqueue_baseline_cached(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    epsilon: Float32,
    has_weight: Int,
    has_bias: Int,
    ctx: DeviceContext,
) raises:
    var cache_name = String(t"LAYER_NORM_FORWARD_F32_BASE_V1_{ctx.id()}")
    comptime FuncT = type_of(
        ctx.compile_function[_layer_norm_forward_f32_baseline]()
    )
    if global_ptr := _get_global_or_null(cache_name):
        var cached = global_ptr.value().bitcast[FuncT]()
        ctx.enqueue_function(
            cached[],
            output,
            mean,
            rstd,
            input,
            weight,
            bias,
            rows,
            cols,
            epsilon,
            has_weight,
            has_bias,
            grid_dim=(min(rows, _MAX_GRID),),
            block_dim=(_BLOCK,),
        )
        return
    var compiled = ctx.compile_function[_layer_norm_forward_f32_baseline]()
    var cached = alloc[FuncT](1)
    cached.init_pointee_move(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_name), cached.bitcast[NoneType]()
    )
    ctx.enqueue_function(
        cached[],
        output,
        mean,
        rstd,
        input,
        weight,
        bias,
        rows,
        cols,
        epsilon,
        has_weight,
        has_bias,
        grid_dim=(min(rows, _MAX_GRID),),
        block_dim=(_BLOCK,),
    )


def enqueue_layer_norm_forward_f32(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    epsilon: Float32,
    has_weight: Bool,
    has_bias: Bool,
    ctx: DeviceContext,
) raises:
    if rows <= 0 or cols <= 0:
        return
    var aligned = (Int(output) | Int(input)) % 16 == 0
    if has_weight:
        aligned = aligned and Int(weight) % 16 == 0
    if has_bias:
        aligned = aligned and Int(bias) % 16 == 0
    if aligned and cols % _VEC == 0 and cols <= 2 * _VEC_BLOCK * _VEC:
        if has_weight:
            if has_bias:
                _enqueue_vec4_cached[True, True](
                    output,
                    mean,
                    rstd,
                    input,
                    weight,
                    bias,
                    rows,
                    cols,
                    epsilon,
                    ctx,
                )
            else:
                _enqueue_vec4_cached[True, False](
                    output,
                    mean,
                    rstd,
                    input,
                    weight,
                    bias,
                    rows,
                    cols,
                    epsilon,
                    ctx,
                )
        elif has_bias:
            _enqueue_vec4_cached[False, True](
                output,
                mean,
                rstd,
                input,
                weight,
                bias,
                rows,
                cols,
                epsilon,
                ctx,
            )
        else:
            _enqueue_vec4_cached[False, False](
                output,
                mean,
                rstd,
                input,
                weight,
                bias,
                rows,
                cols,
                epsilon,
                ctx,
            )
        return
    _enqueue_baseline_cached(
        output,
        mean,
        rstd,
        input,
        weight,
        bias,
        rows,
        cols,
        epsilon,
        1 if has_weight else 0,
        1 if has_bias else 0,
        ctx,
    )
