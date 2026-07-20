# ===----------------------------------------------------------------------=== #
# Thin eager-mode bridge for the accepted BF16 fused cross-entropy path.
#
# Device-kernel bodies live in the production kernel module imported below.
# This Python-visible module only unpacks the runtime pointer ABI and enqueues
# on the caller's DeviceContext. It performs no allocation, host read, or
# synchronization.
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.python._cpython import PyObjectPtr, Py_ssize_t

from bf16_cross_entropy_kernels import (
    enqueue_bf16_cross_entropy_backward,
    enqueue_bf16_cross_entropy_forward,
)
from op_utils import (
    _make_ptr,
    _raw_ctx,
    _raw_int,
    _raw_ret_none,
    _spec_unsupported,
)


def _bf16_cross_entropy_forward_go(
    loss_ptr_obj: PyObjectPtr,
    total_weight_ptr_obj: PyObjectPtr,
    row_max_ptr_obj: PyObjectPtr,
    row_logsum_ptr_obj: PyObjectPtr,
    row_loss_scratch_ptr_obj: PyObjectPtr,
    logits_ptr_obj: PyObjectPtr,
    targets_ptr_obj: PyObjectPtr,
    rows_obj: PyObjectPtr,
    classes_obj: PyObjectPtr,
    ignore_index_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var loss = _make_ptr[DType.float32](
        _raw_int(loss_ptr_obj)
    ).as_unsafe_any_origin()
    var total_weight = _make_ptr[DType.float32](
        _raw_int(total_weight_ptr_obj)
    ).as_unsafe_any_origin()
    var row_max = _make_ptr[DType.float32](
        _raw_int(row_max_ptr_obj)
    ).as_unsafe_any_origin()
    var row_logsum = _make_ptr[DType.float32](
        _raw_int(row_logsum_ptr_obj)
    ).as_unsafe_any_origin()
    var row_loss_scratch = _make_ptr[DType.float32](
        _raw_int(row_loss_scratch_ptr_obj)
    ).as_unsafe_any_origin()
    var logits = _make_ptr[DType.bfloat16](
        _raw_int(logits_ptr_obj)
    ).as_unsafe_any_origin()
    var targets = _make_ptr[DType.int64](
        _raw_int(targets_ptr_obj)
    ).as_unsafe_any_origin()
    var ctx = _raw_ctx(device_context_ptr)
    enqueue_bf16_cross_entropy_forward(
        loss,
        total_weight,
        row_max,
        row_logsum,
        row_loss_scratch,
        logits,
        targets,
        _raw_int(rows_obj),
        _raw_int(classes_obj),
        _raw_int(ignore_index_obj),
        ctx,
    )


def _bf16_cross_entropy_forward_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        if nargs != 11:
            raise Error("Bf16CrossEntropyForward expects exactly 11 arguments")
        _bf16_cross_entropy_forward_go(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
            args[8],
            args[9],
            args[10],
        )
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


def _bf16_cross_entropy_backward_go(
    grad_input_ptr_obj: PyObjectPtr,
    grad_output_ptr_obj: PyObjectPtr,
    logits_ptr_obj: PyObjectPtr,
    targets_ptr_obj: PyObjectPtr,
    row_max_ptr_obj: PyObjectPtr,
    row_logsum_ptr_obj: PyObjectPtr,
    total_weight_ptr_obj: PyObjectPtr,
    rows_obj: PyObjectPtr,
    classes_obj: PyObjectPtr,
    ignore_index_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var grad_input = _make_ptr[DType.bfloat16](
        _raw_int(grad_input_ptr_obj)
    ).as_unsafe_any_origin()
    var grad_output = _make_ptr[DType.float32](
        _raw_int(grad_output_ptr_obj)
    ).as_unsafe_any_origin()
    var logits = _make_ptr[DType.bfloat16](
        _raw_int(logits_ptr_obj)
    ).as_unsafe_any_origin()
    var targets = _make_ptr[DType.int64](
        _raw_int(targets_ptr_obj)
    ).as_unsafe_any_origin()
    var row_max = _make_ptr[DType.float32](
        _raw_int(row_max_ptr_obj)
    ).as_unsafe_any_origin()
    var row_logsum = _make_ptr[DType.float32](
        _raw_int(row_logsum_ptr_obj)
    ).as_unsafe_any_origin()
    var total_weight = _make_ptr[DType.float32](
        _raw_int(total_weight_ptr_obj)
    ).as_unsafe_any_origin()
    var ctx = _raw_ctx(device_context_ptr)
    enqueue_bf16_cross_entropy_backward(
        grad_input,
        grad_output,
        logits,
        targets,
        row_max,
        row_logsum,
        total_weight,
        _raw_int(rows_obj),
        _raw_int(classes_obj),
        _raw_int(ignore_index_obj),
        ctx,
    )


def _bf16_cross_entropy_backward_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        if nargs != 11:
            raise Error("Bf16CrossEntropyBackward expects exactly 11 arguments")
        _bf16_cross_entropy_backward_go(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
            args[8],
            args[9],
            args[10],
        )
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


@export
def PyInit_bf16_cross_entropy_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("bf16_cross_entropy_ops")
        b.def_py_c_function(
            _bf16_cross_entropy_backward_dispatcher,
            "Bf16CrossEntropyBackward",
            docstring=(
                "(grad_input_ptr, grad_output_ptr, logits_ptr, targets_ptr,"
                " row_max_ptr, row_logsum_ptr, total_weight_ptr, rows, classes,"
                " ignore_index, context_ptr); BF16 fused cross-entropy backward"
            ),
        )
        b.def_py_c_function(
            _bf16_cross_entropy_forward_dispatcher,
            "Bf16CrossEntropyForward",
            docstring=(
                "(loss_ptr, total_weight_ptr, row_max_ptr, row_logsum_ptr,"
                " row_loss_scratch_ptr, logits_ptr, targets_ptr, rows, classes,"
                " ignore_index, context_ptr); BF16 fused cross-entropy forward"
            ),
        )
        return b.finalize()
    except e:
        abort(t"failed to create bf16_cross_entropy_ops python module: {e}")
