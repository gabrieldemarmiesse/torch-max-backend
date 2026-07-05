# ===----------------------------------------------------------------------=== #
# Shared helpers for the fast eager-mode kernel modules.
#
# Mirrors `max._interpreter_ops.op_utils`: unwrap `max.driver.Buffer` Python
# objects into raw typed pointers, and rebuild the MAX DeviceContext from the
# pointer that `device._device_context_ptr()` hands us on the Python side.
# ===----------------------------------------------------------------------=== #

from std.gpu.host import DeviceContext
from std.memory import OpaquePointer
from std.python import PythonObject


def _get_dtype(buffer: PythonObject) raises -> DType:
    return DType._from_ui8(UInt8(py=buffer.dtype.value)._mlir_value)


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
