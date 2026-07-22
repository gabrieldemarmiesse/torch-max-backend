"""Runtime-dynamic pure-Mojo FP32 foreach clipping kernels.

Norms use a two-stage reduction. A fixed-size chunk is reduced by one block
into caller-owned scratch, then one block per tensor reduces its chunk
partials and writes the ordered scalar result. Multiplication uses the same
runtime chunk descriptors and applies the device-resident scalar in place.
"""

from std.collections import InlineArray
from std.ffi import _get_global_or_null, external_call
from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.primitives import block
from std.math import min, sqrt
from std.memory import alloc

from foreach_clip_contract import (
    FOREACH_CHUNK_ELEMENTS,
    FOREACH_DESC_CAP,
    FOREACH_THREADS,
    ForeachDesc,
)


comptime _VEC = 4
comptime _MUL_THREADS = 128
comptime _MUL_ILP = 8


@always_inline
def _ptr(addr: Int) -> UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin]:
    return UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin](
        unsafe_from_address=addr
    )


@always_inline
def _chunk_bounds(
    descs: InlineArray[ForeachDesc, FOREACH_DESC_CAP],
    desc_count: Int,
    chunk: Int,
) -> Tuple[ForeachDesc, Int, Int]:
    var desc_index = 0
    while desc_index + 1 < desc_count and chunk >= descs[desc_index].chunk_end:
        desc_index += 1
    var desc = descs[desc_index]
    var first_chunk = 0
    if desc_index != 0:
        first_chunk = descs[desc_index - 1].chunk_end
    var begin = (chunk - first_chunk) * FOREACH_CHUNK_ELEMENTS
    return desc, begin, min(begin + FOREACH_CHUNK_ELEMENTS, desc.numel)


@__name("foreach_l2_norm_f32_chunk_partials_v1")
def _norm_chunk_partials(
    descs: InlineArray[ForeachDesc, FOREACH_DESC_CAP],
    desc_count: Int,
    partials_addr: Int,
):
    var chunk = Int(block_idx.x)
    var desc, begin, end = _chunk_bounds(descs, desc_count, chunk)
    var values = _ptr(desc.tensor_addr)
    var accum = SIMD[DType.float32, _VEC](0.0)
    var index = begin + Int(thread_idx.x) * _VEC
    var stride = FOREACH_THREADS * _VEC
    while index + _VEC <= end:
        var value = values.load[width=_VEC, alignment=4](index)
        accum = value.fma(value, accum)
        index += stride

    var scalar_accum = accum.reduce_add()
    # The contract's chunk size is divisible by four, so only the last chunk
    # of a tensor can need this scalar tail.
    index = begin + ((end - begin) // _VEC) * _VEC + Int(thread_idx.x)
    while index < end:
        var value = values[index]
        scalar_accum += value * value
        index += FOREACH_THREADS

    var total = block.sum[block_size=FOREACH_THREADS, broadcast=False](
        scalar_accum
    )
    if thread_idx.x == 0:
        _ptr(partials_addr)[chunk] = total


@__name("foreach_l2_norm_f32_finalize_v1")
def _norm_finalize(
    descs: InlineArray[ForeachDesc, FOREACH_DESC_CAP],
    desc_count: Int,
    partials_addr: Int,
):
    var desc_index = Int(block_idx.x)
    if desc_index >= desc_count:
        return
    var desc = descs[desc_index]
    var begin = 0
    if desc_index != 0:
        begin = descs[desc_index - 1].chunk_end
    var accum = Float32(0.0)
    for chunk in range(
        begin + Int(thread_idx.x), desc.chunk_end, FOREACH_THREADS
    ):
        accum += _ptr(partials_addr)[chunk]
    var total = block.sum[block_size=FOREACH_THREADS, broadcast=False](accum)
    if thread_idx.x == 0:
        _ptr(desc.output_addr)[0] = sqrt(total)


@__name("foreach_mul_tensor_f32_aligned_streaming_ilp8_t128_v8")
def _mul_aligned_streaming_ilp8_t128(
    descs: InlineArray[ForeachDesc, FOREACH_DESC_CAP],
    desc_count: Int,
    scalar_addr: Int,
):
    var chunk = Int(block_idx.x)
    var desc, begin, end = _chunk_bounds(descs, desc_count, chunk)
    var values = _ptr(desc.tensor_addr)
    var scalar = _ptr(scalar_addr)[0]
    var tid = Int(thread_idx.x)

    # Each descriptor may start at any four-byte alignment. Peeling at most
    # 31 values makes the first full warp access begin on a 128-byte boundary;
    # all following warp accesses then use exactly four 32-byte sectors instead
    # of five. Since the fixed chunk is a multiple of 128 bytes, the same
    # bounded peel applies independently to every chunk.
    var address = desc.tensor_addr + begin * 4
    var prefix = ((128 - address % 128) % 128) // 4
    var body_begin = min(begin + prefix, end)
    var prefix_index = begin + tid
    if prefix_index < body_begin:
        var prefix_value = values.load[width=1, alignment=4, non_temporal=True](
            prefix_index
        )[0]
        values.store[width=1, alignment=4, non_temporal=True](
            prefix_index, SIMD[DType.float32, 1](prefix_value * scalar)
        )

    var index = body_begin + tid
    var stride = _MUL_THREADS * _MUL_ILP
    while index + (_MUL_ILP - 1) * _MUL_THREADS < end:
        var loaded = SIMD[DType.float32, _MUL_ILP]()
        comptime for u in range(_MUL_ILP):
            loaded[u] = values.load[
                width=1,
                alignment=4,
                non_temporal=True,
            ](
                index + u * _MUL_THREADS
            )[0]
        loaded *= scalar
        comptime for u in range(_MUL_ILP):
            values.store[width=1, alignment=4, non_temporal=True](
                index + u * _MUL_THREADS,
                SIMD[DType.float32, 1](loaded[u]),
            )
        index += stride

    while index < end:
        var value = values.load[
            width=1,
            alignment=4,
            non_temporal=True,
        ](
            index
        )[0]
        values.store[width=1, alignment=4, non_temporal=True](
            index, SIMD[DType.float32, 1](value * scalar)
        )
        index += _MUL_THREADS


def _enqueue_norm_partials_cached(
    descs: InlineArray[ForeachDesc, FOREACH_DESC_CAP],
    desc_count: Int,
    total_chunks: Int,
    partials_addr: Int,
    ctx: DeviceContext,
) raises:
    var cache_name = String(t"FOREACH_NORM_PARTIALS_F32_V1_{ctx.id()}")
    comptime FuncT = type_of(ctx.compile_function[_norm_chunk_partials]())
    if global_ptr := _get_global_or_null(cache_name):
        var cached = global_ptr.value().bitcast[FuncT]()
        ctx.enqueue_function(
            cached[],
            descs,
            desc_count,
            partials_addr,
            grid_dim=(total_chunks,),
            block_dim=(FOREACH_THREADS,),
        )
        return
    var compiled = ctx.compile_function[_norm_chunk_partials]()
    var cached = alloc[FuncT](1)
    cached.init_pointee_move(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_name), cached.bitcast[NoneType]()
    )
    ctx.enqueue_function(
        cached[],
        descs,
        desc_count,
        partials_addr,
        grid_dim=(total_chunks,),
        block_dim=(FOREACH_THREADS,),
    )


def _enqueue_norm_finalize_cached(
    descs: InlineArray[ForeachDesc, FOREACH_DESC_CAP],
    desc_count: Int,
    partials_addr: Int,
    ctx: DeviceContext,
) raises:
    var cache_name = String(t"FOREACH_NORM_FINALIZE_F32_V1_{ctx.id()}")
    comptime FuncT = type_of(ctx.compile_function[_norm_finalize]())
    if global_ptr := _get_global_or_null(cache_name):
        var cached = global_ptr.value().bitcast[FuncT]()
        ctx.enqueue_function(
            cached[],
            descs,
            desc_count,
            partials_addr,
            grid_dim=(desc_count,),
            block_dim=(FOREACH_THREADS,),
        )
        return
    var compiled = ctx.compile_function[_norm_finalize]()
    var cached = alloc[FuncT](1)
    cached.init_pointee_move(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_name), cached.bitcast[NoneType]()
    )
    ctx.enqueue_function(
        cached[],
        descs,
        desc_count,
        partials_addr,
        grid_dim=(desc_count,),
        block_dim=(FOREACH_THREADS,),
    )


def _enqueue_mul_cached(
    descs: InlineArray[ForeachDesc, FOREACH_DESC_CAP],
    desc_count: Int,
    total_chunks: Int,
    scalar_addr: Int,
    ctx: DeviceContext,
) raises:
    var cache_name = String(t"FOREACH_MUL_TENSOR_F32_V1_{ctx.id()}")
    comptime FuncT = type_of(
        ctx.compile_function[_mul_aligned_streaming_ilp8_t128]()
    )
    if global_ptr := _get_global_or_null(cache_name):
        var cached = global_ptr.value().bitcast[FuncT]()
        ctx.enqueue_function(
            cached[],
            descs,
            desc_count,
            scalar_addr,
            grid_dim=(total_chunks,),
            block_dim=(_MUL_THREADS,),
        )
        return
    var compiled = ctx.compile_function[_mul_aligned_streaming_ilp8_t128]()
    var cached = alloc[FuncT](1)
    cached.init_pointee_move(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_name), cached.bitcast[NoneType]()
    )
    ctx.enqueue_function(
        cached[],
        descs,
        desc_count,
        scalar_addr,
        grid_dim=(total_chunks,),
        block_dim=(_MUL_THREADS,),
    )


def enqueue_foreach_l2_norm_f32(
    descs: InlineArray[ForeachDesc, FOREACH_DESC_CAP],
    desc_count: Int,
    total_chunks: Int,
    partials_addr: Int,
    ctx: DeviceContext,
) raises:
    if desc_count <= 0:
        return
    if total_chunks > 0:
        _enqueue_norm_partials_cached(
            descs, desc_count, total_chunks, partials_addr, ctx
        )
    _enqueue_norm_finalize_cached(descs, desc_count, partials_addr, ctx)


def enqueue_foreach_mul_tensor_f32(
    descs: InlineArray[ForeachDesc, FOREACH_DESC_CAP],
    desc_count: Int,
    total_chunks: Int,
    scalar_addr: Int,
    ctx: DeviceContext,
) raises:
    if desc_count > 0 and total_chunks > 0:
        _enqueue_mul_cached(descs, desc_count, total_chunks, scalar_addr, ctx)
