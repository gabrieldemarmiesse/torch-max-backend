# ===----------------------------------------------------------------------=== #
# Fast eager-mode reduction kernels for max_device: row-wise sum / max /
# min / argmin / variance / log-softmax over the trailing dimension of a
# contiguous tensor. Reductions over other dim sets are handled on the
# Python side by a zero-copy permute + materialize into row-major layout.
#
# Raw-pointer calling convention (see elementwise_ops.mojo): tensor
# operands arrive as element-aligned int addresses, sizes and dtypes as
# ints, ctx_ptr last. Every kernel has a CPU branch.
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder


@export
def PyInit_reduction_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("reduction_ops")
        return b.finalize()
    except e:
        abort(t"failed to create reduction_ops python module: {e}")
