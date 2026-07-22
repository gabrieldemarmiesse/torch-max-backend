# ===----------------------------------------------------------------------=== #
# Eager multi-GPU collectives for the mojo device, built on Modular's
# pure-Mojo P2P comm kernels (modular repo: max/kernels/src/comm/).
#
# One call launches the collective on EVERY participating GPU: the comm
# kernels are single-process by design — each GPU's kernel instance receives
# peer pointers into the other GPUs' buffers, so all instances must be
# enqueued together from the address space that owns them all. Launches are
# dispatched through `_launch_device_collective`, which places each enqueue
# on the AsyncRT worker with affinity for that device, with the GIL
# released.
#
# Python passes raw data pointers, per-device Signal-buffer pointers and
# DeviceContext pointers (from `Accelerator(i)._device_context_ptr()`), so
# every instance rides its device's default stream and stays ordered with
# the eager kernels and stream-ordered frees on that device.
#
# The Signal buffers are allocated and zeroed on the Python side
# (`torch_mojo_backend/distributed.py`) and must be sized
# `signal_header_bytes() + world_size * payload_nbytes` for the 2-stage
# bandwidth-bound path.
# ===----------------------------------------------------------------------=== #

from std.collections import InlineArray, Optional
from std.memory import OpaquePointer, UnsafePointer
from std.os import abort
from std.gpu.host import DeviceContext, DeviceContextList
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from std.sys import size_of

from comm import MAX_GPUS, Signal
from comm.allreduce import allreduce, elementwise_epilogue_type
from comm.device_collective import _launch_device_collective
from layout import Coord, TileTensor, row_major

# Grad-carrying dtypes DDP buckets can hold.
comptime COMM_DTYPES = [DType.float32, DType.bfloat16, DType.float16]


def signal_header_bytes() raises -> PythonObject:
    """Byte size of the comm Signal header (Python mirrors this to size
    signal buffers as header + world_size * payload bytes)."""
    return PythonObject(size_of[Signal]())


@parameter
def _all_reduce_impl[
    dtype: DType, ngpus: Int
](
    in_ptrs: PythonObject,
    out_ptrs_obj: PythonObject,
    sig_ptrs: PythonObject,
    ctx_ptrs: PythonObject,
    numel: Int,
    average: Bool,
) raises:
    var rank_sigs = InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )
    for i in range(ngpus):
        rank_sigs[i] = UnsafePointer[Signal, MutAnyOrigin](
            unsafe_from_address=Int(py=sig_ptrs[i])
        )

    comptime InTile = TileTensor[dtype, type_of(row_major(0)), ImmutAnyOrigin]
    var in_tiles = InlineArray[InTile, ngpus](uninitialized=True)
    var out_ptrs = InlineArray[
        UnsafePointer[Scalar[dtype], MutAnyOrigin], ngpus
    ](uninitialized=True)
    var ctx_array = InlineArray[DeviceContext, ngpus](uninitialized=True)
    for i in range(ngpus):
        var in_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
            unsafe_from_address=Int(py=in_ptrs[i])
        )
        in_tiles[i] = TileTensor(in_ptr, row_major(numel))
        out_ptrs[i] = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=Int(py=out_ptrs_obj[i])
        )
        # init_pointee_move prevents DeviceContext.__del__ from dropping a
        # refcount that assigning into the uninitialized slot would trigger.
        (ctx_array.unsafe_ptr() + i).init_pointee_move(
            DeviceContext(
                OpaquePointer[MutUntrackedOrigin](
                    unsafe_from_address=Int(py=ctx_ptrs[i])
                )
            )
        )
    var dev_ctxs = DeviceContextList[ngpus](ctx_array^)
    var inv = 1.0 / Float64(ngpus)

    @always_inline
    def launch[
        index: Int
    ]() raises {
        read in_tiles,
        read rank_sigs,
        read out_ptrs,
        read dev_ctxs,
        read numel,
        read inv,
        read average,
    }:
        var out_tile = TileTensor(out_ptrs[index], row_major(numel))

        @always_inline
        @parameter
        @__copy_capture(out_tile, inv)
        def avg_lambda[
            _dtype: DType, _width: SIMDSize, *, _alignment: Int
        ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
            out_tile.store[width=_width, alignment=_alignment](
                coords, (val * inv.cast[_dtype]()).cast[dtype]()
            )

        if average:
            allreduce[
                ngpus=ngpus,
                output_lambda=Optional[elementwise_epilogue_type](avg_lambda),
            ](in_tiles, out_tile, rank_sigs, dev_ctxs[index])
        else:
            allreduce[ngpus=ngpus](
                in_tiles, out_tile, rank_sigs, dev_ctxs[index]
            )

    # Release the GIL during the blocking TaskGroup wait so other Python
    # threads keep dispatching while the per-device enqueues run.
    with GILReleased(Python()):
        _launch_device_collective[ngpus](
            launch, DeviceContextList[ngpus](copy=dev_ctxs)
        )


def all_reduce(
    in_ptrs: PythonObject,
    out_ptrs: PythonObject,
    sig_ptrs: PythonObject,
    ctx_ptrs: PythonObject,
    numel: PythonObject,
    dtype_value: PythonObject,
    ngpus: PythonObject,
    average: PythonObject,
) raises -> PythonObject:
    """Allreduce across `ngpus` GPUs: out[i] = sum over ranks of in[r].

    All argument tuples are indexed by rank. `in_ptrs`/`out_ptrs` are raw
    element-aligned device addresses of contiguous same-shape tensors;
    outputs must not alias inputs (the latency-bound path writes outputs
    while peers still read inputs). `sig_ptrs` point to zero-initialized
    Signal buffers; `ctx_ptrs` are DeviceContext addresses. With
    `average`, each element is scaled by 1/ngpus in the store epilogue.
    """
    var ngpus_v = Int(py=ngpus)
    var numel_v = Int(py=numel)
    var average_v = Bool(py=average)
    var dtype = DType._from_ui8(UInt8(Int(py=dtype_value))._mlir_value)

    comptime for dt in COMM_DTYPES:
        if dtype == dt:
            comptime for n in range(2, MAX_GPUS + 1):
                if ngpus_v == n:
                    _all_reduce_impl[dt, n](
                        in_ptrs, out_ptrs, sig_ptrs, ctx_ptrs, numel_v, average_v
                    )
                    return Python.none()
            raise Error(
                t"mojo all_reduce: ngpus={ngpus_v} must be in [2, {MAX_GPUS}]"
            )
    raise Error(t"mojo all_reduce: unsupported dtype {dtype}")


@export
def PyInit_comm_ops() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("comm_ops")
        m.def_function[signal_header_bytes]("signal_header_bytes")
        m.def_function[all_reduce]("all_reduce")
        return m.finalize()
    except e:
        abort(t"failed to create comm_ops python module: {e}")
