"""DLPack export for TorchMojoTensor.

TorchMojoTensor's torch-side TensorImpl is a zero-byte meta-backed wrapper,
so torch's built-in `__dlpack__` would export a null pointer. The real
allocation lives behind the Python-side metadata (`_ptr`, `_shape`,
`_dtype`, `_device`, `_holder`). This module builds the `DLManagedTensor`
capsule from that metadata so consumers like `max.driver.Buffer.from_dlpack`
can adopt the memory zero-copy — this is how mojo tensors are fed into
compiled MAX graphs.

Only contiguous tensors are exported (callers materialize first), so the
capsule advertises compact row-major layout (strides=NULL). The capsule
keeps the producing tensor's `_holder` alive until the consumer's deleter
runs, which is the same refcount-based ownership the rest of the eager
backend relies on.
"""

import ctypes

from max.dtype import DType


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))

_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLManagedTensorDeleter),
]

# DLPack type codes (see ATen/dlpack.h): kDLInt=0, kDLUInt=1, kDLFloat=2,
# kDLBfloat=4, kDLBool=6.
_DLPACK_CODE_OF: dict[DType, tuple[int, int]] = {
    DType.bool: (6, 8),
    DType.int8: (0, 8),
    DType.int16: (0, 16),
    DType.int32: (0, 32),
    DType.int64: (0, 64),
    DType.uint8: (1, 8),
    DType.uint16: (1, 16),
    DType.uint32: (1, 32),
    DType.uint64: (1, 64),
    DType.float16: (2, 16),
    DType.float32: (2, 32),
    DType.float64: (2, 64),
    DType.bfloat16: (4, 16),
}

# kDLCPU=1, kDLCUDA=2, kDLMetal=8, kDLROCM=10.
_DLPACK_DEVICE_TYPE_OF = {"cpu": 1, "cuda": 2, "metal": 8, "hip": 10}

_CAPSULE_NAME = b"dltensor"

_pyapi = ctypes.pythonapi
_PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_pyapi.PyCapsule_New.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    _PyCapsule_Destructor,
]
_pyapi.PyCapsule_New.restype = ctypes.py_object
_pyapi.PyCapsule_IsValid.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_pyapi.PyCapsule_IsValid.restype = ctypes.c_int
_pyapi.PyCapsule_GetPointer.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_pyapi.PyCapsule_GetPointer.restype = ctypes.c_void_p

# Every live export, keyed by the DLManagedTensor struct address. The entry
# pins the ctypes structs (which must outlive the capsule) and the producing
# tensor's holder (which must outlive the consumer's use of the memory).
_live_exports: dict[int, tuple] = {}


def _deleter_impl(handle):
    _live_exports.pop(ctypes.addressof(handle.contents), None)


_managed_deleter = _DLManagedTensorDeleter(_deleter_impl)


def _capsule_destructor_impl(capsule_ptr):
    # A consumer that adopted the memory renames the capsule to
    # "used_dltensor" and becomes responsible for calling the deleter; if
    # the capsule dies still named "dltensor" it was never consumed and the
    # export is released here.
    if _pyapi.PyCapsule_IsValid(capsule_ptr, _CAPSULE_NAME):
        addr = _pyapi.PyCapsule_GetPointer(capsule_ptr, _CAPSULE_NAME)
        _live_exports.pop(addr, None)


_capsule_destructor = _PyCapsule_Destructor(_capsule_destructor_impl)


def dlpack_device(device) -> tuple[int, int]:
    """The DLPack (device_type, device_id) pair for a max.driver.Device."""
    if device.label == "cpu":
        return (_DLPACK_DEVICE_TYPE_OF["cpu"], 0)
    device_type = _DLPACK_DEVICE_TYPE_OF.get(device.api)
    if device_type is None:
        raise BufferError(f"Cannot export device {device} via DLPack")
    return (device_type, device.id)


def make_capsule(holder, data_ptr: int, shape, dtype: DType, device):
    """A "dltensor" PyCapsule for a contiguous device allocation.

    `holder` is any Python object whose refcount keeps the allocation
    alive; it is pinned until the consumer's deleter runs.
    """
    code_bits = _DLPACK_CODE_OF.get(dtype)
    if code_bits is None:
        raise BufferError(f"dtype {dtype} is not exportable via DLPack")
    ndim = len(shape)
    shape_arr = (ctypes.c_int64 * ndim)(*shape)
    managed = _DLManagedTensor()
    managed.dl_tensor.data = data_ptr
    managed.dl_tensor.device = _DLDevice(*dlpack_device(device))
    managed.dl_tensor.ndim = ndim
    managed.dl_tensor.dtype = _DLDataType(code_bits[0], code_bits[1], 1)
    managed.dl_tensor.shape = shape_arr
    managed.dl_tensor.strides = None  # compact row-major
    managed.dl_tensor.byte_offset = 0
    managed.manager_ctx = None
    managed.deleter = _managed_deleter
    addr = ctypes.addressof(managed)
    _live_exports[addr] = (managed, shape_arr, holder)
    return _pyapi.PyCapsule_New(addr, _CAPSULE_NAME, _capsule_destructor)
