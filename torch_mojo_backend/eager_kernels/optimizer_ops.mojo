"""Thin eager bridge for runtime-dynamic multi-tensor optimizer kernels.

The Python boundary validates the full mutable ATen contract and passes one
flat tuple of raw tensor metadata. This module packs at most 32 descriptors per
by-value launch and enqueues on the tensors' existing DeviceContext. It does no
allocation, host read, synchronization, or vendor-library call.
"""

from std.collections import InlineArray
from std.math import ceildiv
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.python._cpython import PyObjectPtr, Py_ssize_t

from op_utils import (
    _raw_ctx,
    _raw_int,
    _raw_ret_none,
    _raw_tuple_f64,
    _raw_tuple_int,
    _raw_tuple_len,
    _spec_unsupported,
)
from optimizer_contract import (
    ADAMW_CHUNK_ELEMENTS,
    ADAMW_DESC_CAP,
    AdamWDesc,
    empty_adamw_desc,
)
from optimizer_kernels import enqueue_fused_adamw_f32


comptime _ADAMW_RECORD_FIELDS = 7


def _fused_adamw_go(
    metadata_obj: PyObjectPtr,
    scalars_obj: PyObjectPtr,
    dtype_mode_obj: PyObjectPtr,
    flags_obj: PyObjectPtr,
    lr_ptr_obj: PyObjectPtr,
    grad_scale_ptr_obj: PyObjectPtr,
    found_inf_ptr_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var value_count = _raw_tuple_len(metadata_obj)
    if value_count % _ADAMW_RECORD_FIELDS != 0:
        raise Error("invalid fused AdamW metadata field count")
    if _raw_tuple_len(scalars_obj) != 5:
        raise Error("fused AdamW expects five scalar hyperparameters")
    if _raw_int(dtype_mode_obj) != 0:
        raise Error("fused AdamW currently supports homogeneous float32 state")
    var flags = _raw_int(flags_obj)
    if flags < 0 or flags > 3:
        raise Error("invalid fused AdamW flags")

    var ctx = _raw_ctx(device_context_ptr)
    if ctx.api() == "cpu":
        raise Error("fused AdamW requires a Mojo accelerator device")
    var lr_scalar = Float32(_raw_tuple_f64(scalars_obj, 0))
    var beta1 = Float32(_raw_tuple_f64(scalars_obj, 1))
    var beta2 = Float32(_raw_tuple_f64(scalars_obj, 2))
    var weight_decay = Float32(_raw_tuple_f64(scalars_obj, 3))
    var eps = Float32(_raw_tuple_f64(scalars_obj, 4))
    var amsgrad = flags & 1
    var maximize = (flags >> 1) & 1
    var lr_ptr = _raw_int(lr_ptr_obj)
    var grad_scale_ptr = _raw_int(grad_scale_ptr_obj)
    var found_inf_ptr = _raw_int(found_inf_ptr_obj)

    var record = 0
    var record_count = value_count // _ADAMW_RECORD_FIELDS
    while record < record_count:
        # The complete array is encoded by value, so initialize unused slots.
        var descs = InlineArray[AdamWDesc, ADAMW_DESC_CAP](
            fill=empty_adamw_desc()
        )
        var desc_count = 0
        var total_chunks = 0
        while record < record_count and desc_count < ADAMW_DESC_CAP:
            var base = record * _ADAMW_RECORD_FIELDS
            var numel = _raw_tuple_int(metadata_obj, base + 6)
            record += 1
            if numel < 0:
                raise Error("fused AdamW tensor numel must be nonnegative")
            if numel == 0:
                continue
            total_chunks += ceildiv(numel, ADAMW_CHUNK_ELEMENTS)
            descs[desc_count] = AdamWDesc(
                _raw_tuple_int(metadata_obj, base + 0),
                _raw_tuple_int(metadata_obj, base + 1),
                _raw_tuple_int(metadata_obj, base + 2),
                _raw_tuple_int(metadata_obj, base + 3),
                _raw_tuple_int(metadata_obj, base + 4),
                _raw_tuple_int(metadata_obj, base + 5),
                numel,
                total_chunks,
            )
            desc_count += 1

        if desc_count > 0:
            enqueue_fused_adamw_f32(
                descs,
                desc_count,
                total_chunks,
                lr_scalar,
                lr_ptr,
                beta1,
                beta2,
                weight_decay,
                eps,
                amsgrad,
                maximize,
                grad_scale_ptr,
                found_inf_ptr,
                ctx,
            )


def _fused_adamw_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        if nargs != 8:
            raise Error("FusedAdamW expects exactly eight arguments")
        _fused_adamw_go(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
        )
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


@export
def PyInit_optimizer_ops() abi("C") -> PythonObject:
    try:
        var builder = PythonModuleBuilder("optimizer_ops")
        builder.def_py_c_function(
            _fused_adamw_dispatcher,
            "FusedAdamW",
            docstring=(
                "(metadata, scalars, dtype_mode, flags, lr_ptr, "
                "grad_scale_ptr, found_inf_ptr, context_ptr); fused FP32 AdamW"
            ),
        )
        return builder.finalize()
    except e:
        abort(t"failed to create optimizer_ops python module: {e}")
