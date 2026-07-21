"""Runtime-dynamic pure-Mojo BF16 GELU-forward candidate.

Buffers proven 16-byte aligned use one 32-byte BF16 vector per thread.  The
same kernels handle an aligned tail, or an entirely scalar unaligned range,
without any out-of-bounds access.  GELU widens BF16 lanes to FP32 internally
and rounds once on return.  Device functions are cached per supplied context.
"""

from nn.activations import gelu, gelu_tanh
from std.ffi import _get_global_or_null, external_call
from std.gpu import block_idx, grid_dim, thread_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv
from std.memory import alloc


comptime _BLOCK = 256
comptime _VEC = 16


@__name("gelu_forward_bf16_exact_vec16")
def _gelu_forward_bf16_exact(
    output: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    elements: Int,
    vec_count: Int,
):
    var gid = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    if gid < vec_count:
        var base = gid * _VEC
        var values = input.load[width=_VEC, alignment=16](base)
        output.store[width=_VEC, alignment=16](base, gelu(values))
    var index = vec_count * _VEC + gid
    var stride = Int(grid_dim.x) * _BLOCK
    while index < elements:
        output[index] = gelu(input[index])
        index += stride


@__name("gelu_forward_bf16_tanh_vec16")
def _gelu_forward_bf16_tanh(
    output: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    elements: Int,
    vec_count: Int,
):
    var gid = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    if gid < vec_count:
        var base = gid * _VEC
        var values = input.load[width=_VEC, alignment=16](base)
        output.store[width=_VEC, alignment=16](base, gelu_tanh(values))
    var index = vec_count * _VEC + gid
    var stride = Int(grid_dim.x) * _BLOCK
    while index < elements:
        output[index] = gelu_tanh(input[index])
        index += stride


def _enqueue_exact_cached(
    output: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    elements: Int,
    vec_count: Int,
    blocks: Int,
    ctx: DeviceContext,
) raises:
    var cache_name = String(t"GELU_FORWARD_BF16_EXACT_VEC16_V1_{ctx.id()}")
    comptime FuncT = type_of(ctx.compile_function[_gelu_forward_bf16_exact]())
    if global_ptr := _get_global_or_null(cache_name):
        var cached = global_ptr.value().bitcast[FuncT]()
        ctx.enqueue_function(
            cached[],
            output,
            input,
            elements,
            vec_count,
            grid_dim=(blocks,),
            block_dim=(_BLOCK,),
        )
        return
    var compiled = ctx.compile_function[_gelu_forward_bf16_exact]()
    var cached = alloc[FuncT](1)
    cached.init_pointee_move(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_name), cached.bitcast[NoneType]()
    )
    ctx.enqueue_function(
        cached[],
        output,
        input,
        elements,
        vec_count,
        grid_dim=(blocks,),
        block_dim=(_BLOCK,),
    )


def _enqueue_tanh_cached(
    output: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    elements: Int,
    vec_count: Int,
    blocks: Int,
    ctx: DeviceContext,
) raises:
    var cache_name = String(t"GELU_FORWARD_BF16_TANH_VEC16_V1_{ctx.id()}")
    comptime FuncT = type_of(ctx.compile_function[_gelu_forward_bf16_tanh]())
    if global_ptr := _get_global_or_null(cache_name):
        var cached = global_ptr.value().bitcast[FuncT]()
        ctx.enqueue_function(
            cached[],
            output,
            input,
            elements,
            vec_count,
            grid_dim=(blocks,),
            block_dim=(_BLOCK,),
        )
        return
    var compiled = ctx.compile_function[_gelu_forward_bf16_tanh]()
    var cached = alloc[FuncT](1)
    cached.init_pointee_move(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_name), cached.bitcast[NoneType]()
    )
    ctx.enqueue_function(
        cached[],
        output,
        input,
        elements,
        vec_count,
        grid_dim=(blocks,),
        block_dim=(_BLOCK,),
    )


def enqueue_gelu_forward_bf16(
    output: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    elements: Int,
    tanh_approx: Bool,
    ctx: DeviceContext,
) raises:
    if elements <= 0:
        return
    var aligned = (Int(output) | Int(input)) % 16 == 0
    var vec_count = elements // _VEC if aligned else 0
    var blocks = ceildiv(vec_count, _BLOCK) if vec_count > 0 else ceildiv(
        elements, _BLOCK
    )
    if tanh_approx:
        _enqueue_tanh_cached(output, input, elements, vec_count, blocks, ctx)
    else:
        _enqueue_exact_cached(output, input, elements, vec_count, blocks, ctx)
