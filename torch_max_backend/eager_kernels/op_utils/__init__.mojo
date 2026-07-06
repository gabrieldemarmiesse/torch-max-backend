# ===----------------------------------------------------------------------=== #
# Shared helpers for the fast eager-mode kernel modules.
#
# Mirrors `max._interpreter_ops.op_utils`: unwrap `max.driver.Buffer` Python
# objects into raw typed pointers, and rebuild the MAX DeviceContext from the
# pointer that `device._device_context_ptr()` hands us on the Python side.
# ===----------------------------------------------------------------------=== #

from std.builtin.device_passable import DevicePassable
from std.ffi import _get_global_or_null, external_call
from std.gpu.host import DeviceContext
from std.memory import OpaquePointer, alloc
from std.python import PythonObject


def _get_dtype(buffer: PythonObject) raises -> DType:
    return DType._from_ui8(UInt8(py=buffer.dtype.value)._mlir_value)


@always_inline
def _enqueue_cached[
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    func: def(* args: * declared_arg_types) thin -> None,
    *Ts: DevicePassable,
](
    ctx: DeviceContext,
    key: String,
    gx: Int,
    gy: Int,
    gz: Int,
    threads: Int,
    *args: *Ts,
) raises:
    """Enqueue `func`, compiling it at most once per process and context.

    `ctx.enqueue_function[func]` re-runs `compile_function` on every call
    (~180µs even when the runtime's module cache hits); caching the
    `DeviceFunction` in the process-global registry — the same pattern the
    vendor BLAS handle uses — brings the enqueue cost down to a few µs.
    """
    var name = String(t"TMB_KERNEL_{key}_{ctx.id()}")
    comptime FuncT = type_of(ctx.compile_function[func]())

    if global_ptr := _get_global_or_null(name):
        var fptr = global_ptr.value().bitcast[FuncT]()
        ctx.enqueue_function(
            fptr[], *args, grid_dim=(gx, gy, gz), block_dim=(threads,)
        )
        return

    var compiled = ctx.compile_function[func]()
    var fptr = alloc[FuncT](1)
    fptr.init_pointee_move(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name),
        fptr.bitcast[NoneType](),
    )
    ctx.enqueue_function(
        fptr[], *args, grid_dim=(gx, gy, gz), block_dim=(threads,)
    )


def _get_buffer_ptr[
    dtype: DType
](buffer: PythonObject) raises -> UnsafePointer[
    Scalar[dtype], MutUntrackedOrigin
]:
    return UnsafePointer[Scalar[dtype], MutUntrackedOrigin](
        unsafe_from_address=Int(py=buffer._data_ptr())
    )


@always_inline
def _make_ptr[
    dtype: DType
](addr: Int) -> UnsafePointer[Scalar[dtype], MutUntrackedOrigin]:
    """Create a typed pointer from a raw integer address."""
    return UnsafePointer[Scalar[dtype], MutUntrackedOrigin](
        unsafe_from_address=addr
    )


def _get_size(buffer: PythonObject) raises -> Int:
    return Int(py=buffer.num_elements)


def _get_ctx(device_context_ptr: PythonObject) raises -> DeviceContext:
    var addr = Int(py=device_context_ptr)
    return DeviceContext(
        OpaquePointer[MutUntrackedOrigin](unsafe_from_address=addr)
    )
