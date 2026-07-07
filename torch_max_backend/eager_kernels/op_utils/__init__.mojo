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
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr, Py_ssize_t


# The floating-point dtypes the fast kernels specialize for. Dispatchers loop
# over this at compile time (`comptime for dt in FLOAT_DTYPES`) to pick the
# runtime dtype, which unrolls into the same `if dtype == ...` chain without
# repeating the call site once per dtype.
comptime FLOAT_DTYPES = [DType.float32, DType.float16, DType.bfloat16]


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


# ---------------------------------------------------------------------------
# Raw-CPython argument unpacking for METH_FASTCALL dispatchers
# (`def_py_c_function`). The high-level `def_function` path pays an owning
# PythonObject per argument plus PyNumber round-trips per int — several
# hundred ns per argument. These helpers read the exact types aten_fast.py
# passes (ints, tuples of ints, driver.Buffer objects) directly, with
# borrowed references where possible. No type checking: the Python callers
# are internal and guarantee the shapes.
# ---------------------------------------------------------------------------


@always_inline
def _raw_int(obj: PyObjectPtr) -> Int:
    return Int(Python().cpython().PyLong_AsSsize_t(obj))


@always_inline
def _raw_f64(obj: PyObjectPtr) -> Float64:
    return Float64(Python().cpython().PyFloat_AsDouble(obj))


@always_inline
def _raw_tuple_int(t: PyObjectPtr, i: Int) -> Int:
    # PyTuple_GetItem returns a borrowed reference: no refcount traffic.
    ref cpy = Python().cpython()
    return Int(cpy.PyLong_AsSsize_t(cpy.PyTuple_GetItem(t, i)))


@always_inline
def _raw_addr(buffer: PyObjectPtr) -> Int:
    """buffer._data_ptr() via direct CPython calls."""
    ref cpy = Python().cpython()
    var meth = cpy.PyObject_GetAttrString(buffer, "_data_ptr")
    var addr_obj = cpy.PyObject_CallObject(meth, PyObjectPtr())
    var addr = Int(cpy.PyLong_AsSsize_t(addr_obj))
    cpy.Py_DecRef(addr_obj)
    cpy.Py_DecRef(meth)
    return addr


@always_inline
def _raw_numel(buffer: PyObjectPtr) -> Int:
    ref cpy = Python().cpython()
    var v = cpy.PyObject_GetAttrString(buffer, "num_elements")
    var n = Int(cpy.PyLong_AsSsize_t(v))
    cpy.Py_DecRef(v)
    return n


@always_inline
def _raw_dtype(buffer: PyObjectPtr) -> DType:
    ref cpy = Python().cpython()
    var dt = cpy.PyObject_GetAttrString(buffer, "dtype")
    var val = cpy.PyObject_GetAttrString(dt, "value")
    var v = Int(cpy.PyLong_AsSsize_t(val))
    cpy.Py_DecRef(val)
    cpy.Py_DecRef(dt)
    return DType._from_ui8(UInt8(v)._mlir_value)


@always_inline
def _raw_ctx(ptr_obj: PyObjectPtr) -> DeviceContext:
    return DeviceContext(
        OpaquePointer[MutUntrackedOrigin](unsafe_from_address=_raw_int(ptr_obj))
    )


@always_inline
def _raw_ret_none() -> PyObjectPtr:
    # The Python callers ignore the return value; 0 is an immortal cached
    # small int, so this is refcount-only.
    return Python().cpython().PyLong_FromSsize_t(0)


@always_inline
def _raw_tuple_f64(t: PyObjectPtr, i: Int) -> Float64:
    ref cpy = Python().cpython()
    return Float64(cpy.PyFloat_AsDouble(cpy.PyTuple_GetItem(t, i)))


@always_inline
def _raw_tuple_len(t: PyObjectPtr) -> Int:
    return Int(Python().cpython().PyObject_Length(t))
