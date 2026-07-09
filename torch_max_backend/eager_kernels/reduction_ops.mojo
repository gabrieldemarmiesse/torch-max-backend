# ===----------------------------------------------------------------------=== #
# Fast eager-mode reduction kernels for max_device: row-wise sum / max /
# min / prod / argmin / min-with-index / variance / log-softmax / any / all
# over the trailing dimension of a contiguous tensor. Reductions over other
# dim sets are handled on the Python side by a zero-copy permute + materialize
# into a row-major (rows, cols) layout, so every kernel here only ever sees a
# contiguous (rows, cols) buffer and reduces each row to one output element.
#
# Raw-pointer calling convention (see elementwise_ops.mojo / nn_ops.mojo):
# tensor operands arrive as element-aligned int addresses, sizes and dtypes as
# ints, ctx_ptr last. Every kernel has a CPU branch (one sequential task per
# row) and a GPU branch (one thread block per row, shared-memory tree reduce
# via `_enqueue_cached`) — the same split nn_ops uses for layer norm / softmax
# / argmax, because full reductions and vocab-dim reductions are called with
# rows == 1 and cols in the tens of thousands, where a thread-per-row launch
# would leave the GPU idle.
#
# Floating-point rows accumulate in float32 (matching torch); integer rows
# accumulate in their own dtype.
# ===----------------------------------------------------------------------=== #

from std.builtin.simd_size import SIMDSize
from std.os import abort
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    block_idx,
    thread_idx,
)
from std.gpu.host import DeviceContext
from std.math import exp, log
from std.memory import stack_allocation
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator
from std.utils.coord import Coord
from std.utils.index import IndexList
from std.utils.numerics import min_or_neg_inf, max_or_inf
from std.utils.static_tuple import StaticTuple

from std.algorithm.functional import elementwise
from std.algorithm.reduction import product, sum
from std.algorithm.reduction import max as reduce_max
from std.algorithm.reduction import min as reduce_min

from std.python._cpython import PyObjectPtr, Py_ssize_t

from op_utils import (
    _enqueue_cached,
    _make_ptr,
    _raw_ctx,
    _raw_dtype_int,
    _raw_f64,
    _raw_int,
    _raw_ret_none,
)


@always_inline
def _parallel_for[
    func: def[width: Int, alignment: Int = 1](Coord) capturing[_] -> None
](count: Int, ctx: DeviceContext) raises:
    """Run `func` once per index in [0, count) on the device queue."""
    if ctx.api() == "cpu":
        elementwise[func, simd_width=1](Coord(count), ctx)
    else:
        comptime if has_accelerator():
            elementwise[func, simd_width=1, target="gpu"](Coord(count), ctx)
        else:
            raise Error("no GPU accelerator available at compile time")


comptime ROWRED_THREADS = 256
# log2(ROWRED_THREADS): number of halving steps in the reduction trees.
comptime ROWRED_STAGES = 8

# Reduction opcodes shared by the generic value-reduction kernel.
comptime RED_SUM = 0
comptime RED_MAX = 1
comptime RED_MIN = 2
comptime RED_PROD = 3


@always_inline
def _accum_dtype[dtype: DType]() -> DType:
    """float rows accumulate in float32; int rows in their own dtype."""
    comptime if dtype.is_floating_point():
        return DType.float32
    else:
        return dtype


# ---------------------------------------------------------------------------
# Library-vs-block routing gate (GPU only).
#
# Benchmarking against the pinned nightly showed the stdlib reduction library is
# NOT strictly better than a 256-thread block-per-row kernel: its two-phase tier
# allocates two device buffers + a memset per call (bad for the small full
# reductions the decode loop issues every step, e.g. max(1,256)/all(1,~3000)),
# and its block-saturated tier uses 128 threads/block (half ours), losing ~2x on
# few-row, huge-col shapes like (256, vocab). The library only wins where there
# are too few rows to saturate the device yet each row is huge (its two-phase
# multi-block fan-out) — full reductions of multi-million-element tensors. Route
# only that regime to the library; everything else uses the block kernel below.
# ---------------------------------------------------------------------------

comptime LIB_MIN_COLS = 1 << 20  # 1,048,576
comptime LIB_MAX_ROWS = 128


@always_inline
def _use_library_reduce(rows: Int, cols: Int) -> Bool:
    return rows <= LIB_MAX_ROWS and cols >= LIB_MIN_COLS


@always_inline
def _red_init[acc_dtype: DType, op_code: Int]() -> Scalar[acc_dtype]:
    comptime if op_code == RED_SUM:
        return Scalar[acc_dtype](0)
    comptime if op_code == RED_PROD:
        return Scalar[acc_dtype](1)
    comptime if op_code == RED_MAX:
        return min_or_neg_inf[acc_dtype]()
    comptime if op_code == RED_MIN:
        return max_or_inf[acc_dtype]()
    return Scalar[acc_dtype](0)


@always_inline
def _red_combine[
    acc_dtype: DType, op_code: Int
](mut acc: Scalar[acc_dtype], v: Scalar[acc_dtype]):
    comptime if op_code == RED_SUM:
        acc += v
    comptime if op_code == RED_PROD:
        acc *= v
    comptime if op_code == RED_MAX:
        if v > acc:
            acc = v
    comptime if op_code == RED_MIN:
        if v < acc:
            acc = v


# ---------------------------------------------------------------------------
# Generic row reduction (sum / max / min / prod): input viewed as
# (rows, cols) contiguous, out has `rows` elements of the same dtype.
# rows == 1 covers full reductions (x.sum(), x.max(), ...).
#
# The under-saturated, huge-col case is handed to the stdlib reduction library
# (`std.algorithm.reduction`) via per-element input_fn/output_fn closures; its
# two-phase tier fans the reduction across the device (rows == 1 over millions
# of elements: 24 ms -> 0.2 ms). Every other GPU case, and all CPU cases, use a
# 256-thread block-per-row kernel (no per-call allocation). Floats reduce in
# float32 (`_accum_dtype`), matching torch.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROWRED_THREADS))
)
@__name(t"reduce_rows_block_{dtype}_{op_code}")
def _reduce_block_kernel[
    dtype: DType, op_code: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cols: Int,
):
    """One block per row (grid.x = rows); lanes stride over the row and
    tree-reduce their partials in shared memory."""
    comptime acc_dtype = _accum_dtype[dtype]()
    var r = block_idx.x
    var tid = thread_idx.x
    var base = r * cols

    var acc = _red_init[acc_dtype, op_code]()
    for j in range(tid, cols, ROWRED_THREADS):
        var v = in_ptr[base + j].cast[acc_dtype]()
        _red_combine[acc_dtype, op_code](acc, v)

    var red = stack_allocation[
        ROWRED_THREADS, acc_dtype, address_space=AddressSpace.SHARED
    ]()
    red[tid] = acc
    barrier()
    var stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            var cur = red[tid]
            _red_combine[acc_dtype, op_code](cur, red[tid + stride])
            red[tid] = cur
        barrier()
        stride //= 2
    if tid == 0:
        out_ptr[r] = red[0].cast[dtype]()


@always_inline
def _reduce_rows[
    dtype: DType, op_code: Int
](out_addr: Int, in_addr: Int, rows: Int, cols: Int, ctx: DeviceContext) raises:
    comptime acc_dtype = _accum_dtype[dtype]()
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @parameter
    @__copy_capture(in_ptr)
    def input_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[acc_dtype, width]:
        var flat = coords[0] * cols + coords[1]
        return in_ptr.load[width=width](flat).cast[acc_dtype]()

    @always_inline
    @parameter
    @__copy_capture(out_ptr)
    def output_fn[
        width: SIMDSize, rank: Int
    ](coords: IndexList[rank], val: SIMD[acc_dtype, width]):
        out_ptr[coords[0]] = val[0].cast[dtype]()

    var shape = IndexList[2](rows, cols)

    @always_inline
    @parameter
    def run[target: StaticString]() raises:
        comptime if op_code == RED_SUM:
            sum[acc_dtype, input_fn, output_fn, target=target, reduce_dim=1](
                Coord(shape), ctx
            )
        elif op_code == RED_PROD:
            product[
                acc_dtype, input_fn, output_fn, target=target, reduce_dim=1
            ](Coord(shape), ctx)
        elif op_code == RED_MAX:
            reduce_max[
                acc_dtype, input_fn, output_fn, target=target, reduce_dim=1
            ](Coord(shape), ctx)
        else:  # RED_MIN
            reduce_min[
                acc_dtype, input_fn, output_fn, target=target, reduce_dim=1
            ](Coord(shape), ctx)

    if ctx.api() == "cpu":
        run["cpu"]()
    else:
        comptime if has_accelerator():
            if _use_library_reduce(rows, cols):
                run["gpu"]()
            else:
                _enqueue_cached[_reduce_block_kernel[dtype, op_code]](
                    ctx,
                    String(t"reduce_rows_{dtype}_{op_code}"),
                    rows,
                    1,
                    1,
                    ROWRED_THREADS,
                    out_ptr.as_unsafe_any_origin(),
                    in_ptr.as_unsafe_any_origin().as_immutable(),
                    cols,
                )
        else:
            raise Error("no GPU accelerator available at compile time")


@always_inline
def _reduce_rows_go[
    op_code: Int
](
    out_ptr_obj: PyObjectPtr,
    in_ptr_obj: PyObjectPtr,
    rows: PyObjectPtr,
    cols: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var out_addr = _raw_int(out_ptr_obj)
    var in_addr = _raw_int(in_ptr_obj)
    var rows_val = _raw_int(rows)
    var cols_val = _raw_int(cols)
    var ctx = _raw_ctx(device_context_ptr)

    # The stdlib GPU reduction warp-shuffles the accumulator, which supports
    # float32/float16/bfloat16/int32/int64 but not sub-32-bit ints. aten only
    # ever routes sum/amax/amin/min here with these dtypes: bool/uint8 sums are
    # promoted to int64 upstream in fast_aten_sum (they are in _CAST_DTYPES),
    # while int8/int16 return NOT_HANDLED there and never reach this kernel. So
    # the narrower list loses no reachable coverage.
    var handled = False
    comptime for dt in [
        DType.float32,
        DType.float16,
        DType.bfloat16,
        DType.int64,
        DType.int32,
    ]:
        if dtype == dt:
            _reduce_rows[dt, op_code](
                out_addr, in_addr, rows_val, cols_val, ctx
            )
            handled = True
    if not handled:
        raise Error("unsupported dtype for fast reduce: " + String(dtype))


# ---------------------------------------------------------------------------
# Row-wise argmin: input viewed as (rows, cols), out is `rows` int64 indices
# (first occurrence wins, matching torch). Mirror of nn_ops' ArgmaxRows with
# the comparison flipped.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROWRED_THREADS))
)
@__name(t"argmin_rows_block_{dtype}")
def _argmin_rows_block_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cols: Int,
):
    """One block per row; lanes pick their own first-min (value, index) with
    strict `<`, then a shared-memory tree reduction combines lanes with a
    lower-index tiebreak on equal values — torch's first-occurrence-wins."""
    var r = block_idx.x
    var tid = thread_idx.x
    var base = r * cols

    var best_val = max_or_inf[dtype]()
    var best_idx = Int64(-1)
    for j in range(tid, cols, ROWRED_THREADS):
        var v = in_ptr[base + j]
        if v < best_val:
            best_val = v
            best_idx = Int64(j)

    var val_smem = stack_allocation[
        ROWRED_THREADS, dtype, address_space=AddressSpace.SHARED
    ]()
    var idx_smem = stack_allocation[
        ROWRED_THREADS, DType.int64, address_space=AddressSpace.SHARED
    ]()
    val_smem[tid] = best_val
    idx_smem[tid] = best_idx
    barrier()

    var stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            var other_val = val_smem[tid + stride]
            var other_idx = idx_smem[tid + stride]
            var cur_val = val_smem[tid]
            var cur_idx = idx_smem[tid]
            if other_val < cur_val or (
                other_val == cur_val
                and other_idx != Int64(-1)
                and (cur_idx == Int64(-1) or other_idx < cur_idx)
            ):
                val_smem[tid] = other_val
                idx_smem[tid] = other_idx
        barrier()
        stride //= 2

    if tid == 0:
        out_ptr[r] = idx_smem[0]


@always_inline
def _argmin_rows[
    dtype: DType
](out_addr: Int, in_addr: Int, rows: Int, cols: Int, ctx: DeviceContext) raises:
    var out_ptr = _make_ptr[DType.int64](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var r = Int(idx[0].value())
            var base = r * cols
            var best = in_ptr[base]
            var best_idx = 0
            for j in range(1, cols):
                var v = in_ptr[base + j]
                if v < best:
                    best = v
                    best_idx = j
            out_ptr[r] = Int64(best_idx)

        _parallel_for[func](rows, ctx)
    else:
        comptime if has_accelerator():
            _enqueue_cached[_argmin_rows_block_kernel[dtype]](
                ctx,
                String(t"argmin_rows_{dtype}"),
                rows,
                1,
                1,
                ROWRED_THREADS,
                out_ptr.as_unsafe_any_origin(),
                in_ptr.as_unsafe_any_origin().as_immutable(),
                cols,
            )
        else:
            raise Error("no GPU accelerator available at compile time")


def _argmin_rows_go(
    out_ptr_obj: PyObjectPtr,
    in_ptr_obj: PyObjectPtr,
    rows: PyObjectPtr,
    cols: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var out_addr = _raw_int(out_ptr_obj)
    var in_addr = _raw_int(in_ptr_obj)
    var rows_val = _raw_int(rows)
    var cols_val = _raw_int(cols)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in [
        DType.float32,
        DType.float16,
        DType.bfloat16,
        DType.int64,
        DType.int32,
    ]:
        if dtype == dt:
            _argmin_rows[dt](out_addr, in_addr, rows_val, cols_val, ctx)
            handled = True
    if not handled:
        raise Error("unsupported dtype for fast argmin: " + String(dtype))


# ---------------------------------------------------------------------------
# Row-wise min/max with indices: values AND int64 indices (first occurrence
# wins). `is_min` selects the direction. Covers aten.min.dim / max.dim.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROWRED_THREADS))
)
@__name(t"minmax_idx_rows_block_{dtype}_{is_min}")
def _minmax_idx_block_kernel[
    dtype: DType, is_min: Bool
](
    val_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    idx_ptr: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cols: Int,
):
    var r = block_idx.x
    var tid = thread_idx.x
    var base = r * cols

    var best_val = max_or_inf[dtype]() if is_min else min_or_neg_inf[dtype]()
    var best_idx = Int64(-1)
    for j in range(tid, cols, ROWRED_THREADS):
        var v = in_ptr[base + j]
        var take = (v < best_val) if is_min else (v > best_val)
        if take:
            best_val = v
            best_idx = Int64(j)

    var val_smem = stack_allocation[
        ROWRED_THREADS, dtype, address_space=AddressSpace.SHARED
    ]()
    var idx_smem = stack_allocation[
        ROWRED_THREADS, DType.int64, address_space=AddressSpace.SHARED
    ]()
    val_smem[tid] = best_val
    idx_smem[tid] = best_idx
    barrier()

    var stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            var other_val = val_smem[tid + stride]
            var other_idx = idx_smem[tid + stride]
            var cur_val = val_smem[tid]
            var cur_idx = idx_smem[tid]
            var strictly_better = (other_val < cur_val) if is_min else (
                other_val > cur_val
            )
            if strictly_better or (
                other_val == cur_val
                and other_idx != Int64(-1)
                and (cur_idx == Int64(-1) or other_idx < cur_idx)
            ):
                val_smem[tid] = other_val
                idx_smem[tid] = other_idx
        barrier()
        stride //= 2

    if tid == 0:
        val_ptr[r] = val_smem[0]
        idx_ptr[r] = idx_smem[0]


@always_inline
def _minmax_idx_rows[
    dtype: DType, is_min: Bool
](
    val_addr: Int,
    idx_addr: Int,
    in_addr: Int,
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) raises:
    var val_ptr = _make_ptr[dtype](val_addr)
    var idx_ptr = _make_ptr[DType.int64](idx_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(val_ptr, idx_ptr, in_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var r = Int(idx[0].value())
            var base = r * cols
            var best = in_ptr[base]
            var best_idx = 0
            for j in range(1, cols):
                var v = in_ptr[base + j]
                var take = (v < best) if is_min else (v > best)
                if take:
                    best = v
                    best_idx = j
            val_ptr[r] = best
            idx_ptr[r] = Int64(best_idx)

        _parallel_for[func](rows, ctx)
    else:
        comptime if has_accelerator():
            _enqueue_cached[_minmax_idx_block_kernel[dtype, is_min]](
                ctx,
                String(t"minmax_idx_rows_{dtype}_{is_min}"),
                rows,
                1,
                1,
                ROWRED_THREADS,
                val_ptr.as_unsafe_any_origin(),
                idx_ptr.as_unsafe_any_origin(),
                in_ptr.as_unsafe_any_origin().as_immutable(),
                cols,
            )
        else:
            raise Error("no GPU accelerator available at compile time")


@always_inline
def _minmax_idx_dispatch[
    is_min: Bool
](
    dtype: DType,
    val_addr: Int,
    idx_addr: Int,
    in_addr: Int,
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) raises:
    var handled = False
    comptime for dt in [
        DType.float32,
        DType.float16,
        DType.bfloat16,
        DType.int64,
        DType.int32,
    ]:
        if dtype == dt:
            _minmax_idx_rows[dt, is_min](
                val_addr, idx_addr, in_addr, rows, cols, ctx
            )
            handled = True
    if not handled:
        raise Error("unsupported dtype for fast min/max.dim: " + String(dtype))


def _minmax_idx_rows_go(
    val_ptr_obj: PyObjectPtr,
    idx_ptr_obj: PyObjectPtr,
    in_ptr_obj: PyObjectPtr,
    rows: PyObjectPtr,
    cols: PyObjectPtr,
    is_min: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var val_addr = _raw_int(val_ptr_obj)
    var idx_addr = _raw_int(idx_ptr_obj)
    var in_addr = _raw_int(in_ptr_obj)
    var rows_val = _raw_int(rows)
    var cols_val = _raw_int(cols)
    var is_min_val = _raw_int(is_min)
    var ctx = _raw_ctx(device_context_ptr)

    if is_min_val != 0:
        _minmax_idx_dispatch[True](
            dtype, val_addr, idx_addr, in_addr, rows_val, cols_val, ctx
        )
    else:
        _minmax_idx_dispatch[False](
            dtype, val_addr, idx_addr, in_addr, rows_val, cols_val, ctx
        )


# ---------------------------------------------------------------------------
# Row-wise variance (two-pass mean then squared deviations, float32 accum),
# divided by (cols - correction). Covers aten.var.correction.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROWRED_THREADS))
)
@__name(t"var_rows_block_{dtype}")
def _var_rows_block_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cols: Int,
    correction: Float32,
):
    var r = block_idx.x
    var tid = thread_idx.x
    var base = r * cols

    var red = stack_allocation[
        ROWRED_THREADS, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var bcast = stack_allocation[
        1, DType.float32, address_space=AddressSpace.SHARED
    ]()

    var s = Float32(0)
    for j in range(tid, cols, ROWRED_THREADS):
        s += in_ptr[base + j].cast[DType.float32]()
    red[tid] = s
    barrier()
    var stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            red[tid] += red[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        bcast[0] = red[0] / Float32(cols)
    barrier()
    var mean = bcast[0]

    var vs = Float32(0)
    for j in range(tid, cols, ROWRED_THREADS):
        var d = in_ptr[base + j].cast[DType.float32]() - mean
        vs += d * d
    red[tid] = vs
    barrier()
    stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            red[tid] += red[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        out_ptr[r] = (red[0] / (Float32(cols) - correction)).cast[dtype]()


@always_inline
def _var_rows[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    rows: Int,
    cols: Int,
    correction: Float32,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var r = Int(idx[0].value())
            var base = r * cols
            var total = Float32(0)
            for j in range(cols):
                total += in_ptr[base + j].cast[DType.float32]()
            var mean = total / Float32(cols)
            var var_sum = Float32(0)
            for j in range(cols):
                var d = in_ptr[base + j].cast[DType.float32]() - mean
                var_sum += d * d
            out_ptr[r] = (var_sum / (Float32(cols) - correction)).cast[dtype]()

        _parallel_for[func](rows, ctx)
    else:
        comptime if has_accelerator():
            _enqueue_cached[_var_rows_block_kernel[dtype]](
                ctx,
                String(t"var_rows_{dtype}"),
                rows,
                1,
                1,
                ROWRED_THREADS,
                out_ptr.as_unsafe_any_origin(),
                in_ptr.as_unsafe_any_origin().as_immutable(),
                cols,
                correction,
            )
        else:
            raise Error("no GPU accelerator available at compile time")


def _var_rows_go(
    out_ptr_obj: PyObjectPtr,
    in_ptr_obj: PyObjectPtr,
    rows: PyObjectPtr,
    cols: PyObjectPtr,
    correction: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var out_addr = _raw_int(out_ptr_obj)
    var in_addr = _raw_int(in_ptr_obj)
    var rows_val = _raw_int(rows)
    var cols_val = _raw_int(cols)
    var correction_val = Float32(_raw_f64(correction))
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in [DType.float32, DType.float16, DType.bfloat16]:
        if dtype == dt:
            _var_rows[dt](
                out_addr, in_addr, rows_val, cols_val, correction_val, ctx
            )
            handled = True
    if not handled:
        raise Error("unsupported dtype for fast var: " + String(dtype))


# ---------------------------------------------------------------------------
# Row-wise log-softmax over the trailing dim: out = x - max - log(sum(exp(x -
# max))), float32 accumulation. Covers aten._log_softmax.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROWRED_THREADS))
)
@__name(t"log_softmax_rows_block_{dtype}")
def _log_softmax_rows_block_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cols: Int,
):
    var r = block_idx.x
    var tid = thread_idx.x
    var base = r * cols

    var red = stack_allocation[
        ROWRED_THREADS, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var bcast = stack_allocation[
        2, DType.float32, address_space=AddressSpace.SHARED
    ]()

    var m = Float32.MIN
    for j in range(tid, cols, ROWRED_THREADS):
        var x = in_ptr[base + j].cast[DType.float32]()
        if x > m:
            m = x
    red[tid] = m
    barrier()
    var stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            if red[tid + stride] > red[tid]:
                red[tid] = red[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        bcast[0] = red[0]
    barrier()
    m = bcast[0]

    var s = Float32(0)
    for j in range(tid, cols, ROWRED_THREADS):
        s += exp(in_ptr[base + j].cast[DType.float32]() - m)
    red[tid] = s
    barrier()
    stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            red[tid] += red[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        bcast[1] = log(red[0])
    barrier()
    var log_denom = bcast[1]

    for j in range(tid, cols, ROWRED_THREADS):
        var x = in_ptr[base + j].cast[DType.float32]()
        out_ptr[base + j] = (x - m - log_denom).cast[dtype]()


@always_inline
def _log_softmax_rows[
    dtype: DType
](out_addr: Int, in_addr: Int, rows: Int, cols: Int, ctx: DeviceContext) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var r = Int(idx[0].value())
            var base = r * cols
            var m = Float32.MIN
            for j in range(cols):
                var x = in_ptr[base + j].cast[DType.float32]()
                if x > m:
                    m = x
            var denom = Float32(0)
            for j in range(cols):
                denom += exp(in_ptr[base + j].cast[DType.float32]() - m)
            var log_denom = log(denom)
            for j in range(cols):
                var x = in_ptr[base + j].cast[DType.float32]()
                out_ptr[base + j] = (x - m - log_denom).cast[dtype]()

        _parallel_for[func](rows, ctx)
    else:
        comptime if has_accelerator():
            _enqueue_cached[_log_softmax_rows_block_kernel[dtype]](
                ctx,
                String(t"log_softmax_rows_{dtype}"),
                rows,
                1,
                1,
                ROWRED_THREADS,
                out_ptr.as_unsafe_any_origin(),
                in_ptr.as_unsafe_any_origin().as_immutable(),
                cols,
            )
        else:
            raise Error("no GPU accelerator available at compile time")


def _log_softmax_rows_go(
    out_ptr_obj: PyObjectPtr,
    in_ptr_obj: PyObjectPtr,
    rows: PyObjectPtr,
    cols: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var out_addr = _raw_int(out_ptr_obj)
    var in_addr = _raw_int(in_ptr_obj)
    var rows_val = _raw_int(rows)
    var cols_val = _raw_int(cols)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in [DType.float32, DType.float16, DType.bfloat16]:
        if dtype == dt:
            _log_softmax_rows[dt](out_addr, in_addr, rows_val, cols_val, ctx)
            handled = True
    if not handled:
        raise Error("unsupported dtype for fast log_softmax: " + String(dtype))


# ---------------------------------------------------------------------------
# Row-wise any / all: input viewed as (rows, cols) of any dtype, out is
# `rows` bool elements (nonzero test). Covers aten.any.dim / all.dim(s).
# Same library-vs-block routing as the value reductions: the library's max/min
# accept DType.bool natively (any = max init False -> OR; all = min init True ->
# AND), but its two-phase tier allocates per call, so only the under-saturated,
# huge-col regime uses it; everything else uses the block kernel.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROWRED_THREADS))
)
@__name(t"anyall_rows_block_{dtype}_{is_all}")
def _anyall_rows_block_kernel[
    dtype: DType, is_all: Bool
](
    out_ptr: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cols: Int,
):
    var r = block_idx.x
    var tid = thread_idx.x
    var base = r * cols
    var zero = Scalar[dtype](0)

    var acc: Bool = is_all
    for j in range(tid, cols, ROWRED_THREADS):
        var nz = Bool(in_ptr[base + j] != zero)
        comptime if is_all:
            acc = acc and nz
        else:
            acc = acc or nz

    var red = stack_allocation[
        ROWRED_THREADS, DType.bool, address_space=AddressSpace.SHARED
    ]()
    red[tid] = acc
    barrier()
    var stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            comptime if is_all:
                red[tid] = red[tid] and red[tid + stride]
            else:
                red[tid] = red[tid] or red[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        out_ptr[r] = red[0]


@always_inline
def _anyall_rows[
    dtype: DType, is_all: Bool
](out_addr: Int, in_addr: Int, rows: Int, cols: Int, ctx: DeviceContext) raises:
    """any/all over the trailing axis. The nonzero test maps each row to bools,
    then any = max / all = min over DType.bool. Routes the under-saturated,
    huge-col regime to the stdlib library and everything else to a block kernel
    (see `_use_library_reduce`)."""
    var out_ptr = _make_ptr[DType.bool](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    # cols == 0 always arrives with rows == 0 (an empty reduce dim means an
    # empty tensor; the guard lives upstream in aten_fast._reduce_to_rows), so
    # both the library (zero-size shape check) and the block kernel (grid 0)
    # write nothing — same pre-existing contract as the old kernels.

    @always_inline
    @parameter
    @__copy_capture(in_ptr)
    def input_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[DType.bool, width]:
        var flat = coords[0] * cols + coords[1]
        # Nonzero test with the block kernel's `!=` semantics: `.eq` is an
        # ORDERED equal (NaN == 0 -> False), so ~eq gives NaN -> True, matching
        # torch's NaN-is-truthy any/all. (`.cast[DType.bool]()` lowers to an
        # ordered ne, which would wrongly map NaN -> False.) For bool input,
        # eq-with-False then invert is the identity.
        var v = in_ptr.load[width=width](flat)
        return ~v.eq(SIMD[dtype, width]())

    @always_inline
    @parameter
    @__copy_capture(out_ptr)
    def output_fn[
        width: SIMDSize, rank: Int
    ](coords: IndexList[rank], val: SIMD[DType.bool, width]):
        out_ptr[coords[0]] = val[0]

    var shape = IndexList[2](rows, cols)

    @always_inline
    @parameter
    def run[target: StaticString]() raises:
        comptime if is_all:
            reduce_min[
                DType.bool, input_fn, output_fn, target=target, reduce_dim=1
            ](Coord(shape), ctx)
        else:
            reduce_max[
                DType.bool, input_fn, output_fn, target=target, reduce_dim=1
            ](Coord(shape), ctx)

    if ctx.api() == "cpu":
        run["cpu"]()
    else:
        comptime if has_accelerator():
            if _use_library_reduce(rows, cols):
                run["gpu"]()
            else:
                _enqueue_cached[_anyall_rows_block_kernel[dtype, is_all]](
                    ctx,
                    String(t"anyall_rows_{dtype}_{is_all}"),
                    rows,
                    1,
                    1,
                    ROWRED_THREADS,
                    out_ptr.as_unsafe_any_origin(),
                    in_ptr.as_unsafe_any_origin().as_immutable(),
                    cols,
                )
        else:
            raise Error("no GPU accelerator available at compile time")


@always_inline
def _anyall_rows_dispatch[
    is_all: Bool
](
    dtype: DType,
    out_addr: Int,
    in_addr: Int,
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) raises:
    var handled = False
    comptime for dt in [
        DType.float32,
        DType.float16,
        DType.bfloat16,
        DType.int64,
        DType.int32,
        DType.int16,
        DType.int8,
        DType.uint8,
        DType.bool,
    ]:
        if dtype == dt:
            _anyall_rows[dt, is_all](out_addr, in_addr, rows, cols, ctx)
            handled = True
    if not handled:
        raise Error("unsupported dtype for fast any/all: " + String(dtype))


def _anyall_rows_go[
    is_all: Bool
](
    out_ptr_obj: PyObjectPtr,
    in_ptr_obj: PyObjectPtr,
    rows: PyObjectPtr,
    cols: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    _anyall_rows_dispatch[is_all](
        _raw_dtype_int(dtype_obj),
        _raw_int(out_ptr_obj),
        _raw_int(in_ptr_obj),
        _raw_int(rows),
        _raw_int(cols),
        _raw_ctx(device_context_ptr),
    )


# ---------------------------------------------------------------------------
# METH_FASTCALL wrappers: raw CPython argument unpacking (no owning
# PythonObject per argument). Argument types are guaranteed by the internal
# Python callers; raise sites are unsupported-dtype guards gated upstream.
# ---------------------------------------------------------------------------


def _reduce_dispatcher[
    op_code: Int
](
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _reduce_rows_go[op_code](
            args[0], args[1], args[2], args[3], args[4], args[5]
        )
    except:
        pass
    return _raw_ret_none()


def _argmin_rows_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _argmin_rows_go(args[0], args[1], args[2], args[3], args[4], args[5])
    except:
        pass
    return _raw_ret_none()


def _minmax_idx_rows_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _minmax_idx_rows_go(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
        )
    except:
        pass
    return _raw_ret_none()


def _var_rows_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _var_rows_go(
            args[0], args[1], args[2], args[3], args[4], args[5], args[6]
        )
    except:
        pass
    return _raw_ret_none()


def _log_softmax_rows_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _log_softmax_rows_go(
            args[0], args[1], args[2], args[3], args[4], args[5]
        )
    except:
        pass
    return _raw_ret_none()


def _anyall_rows_dispatcher[
    is_all: Bool
](
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _anyall_rows_go[is_all](
            args[0], args[1], args[2], args[3], args[4], args[5]
        )
    except:
        pass
    return _raw_ret_none()


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_reduction_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("reduction_ops")
        b.def_py_c_function(
            _reduce_dispatcher[RED_SUM],
            "SumRows",
            docstring="sum over the last dim (rows, cols) -> (rows,)",
        )
        b.def_py_c_function(
            _reduce_dispatcher[RED_MAX],
            "MaxRowsR",
            docstring="max over the last dim (rows, cols) -> (rows,)",
        )
        b.def_py_c_function(
            _reduce_dispatcher[RED_MIN],
            "MinRows",
            docstring="min over the last dim (rows, cols) -> (rows,)",
        )
        b.def_py_c_function(
            _reduce_dispatcher[RED_PROD],
            "ProdRows",
            docstring="prod over the last dim (rows, cols) -> (rows,)",
        )
        b.def_py_c_function(
            _argmin_rows_dispatcher,
            "ArgminRows",
            docstring="argmin over the last dim (rows, cols) -> int64 (rows,)",
        )
        b.def_py_c_function(
            _minmax_idx_rows_dispatcher,
            "MinMaxIdxRows",
            docstring=(
                "min/max over the last dim with int64 indices (rows, cols) ->"
                " values (rows,), indices (rows,); is_min flag picks direction"
            ),
        )
        b.def_py_c_function(
            _var_rows_dispatcher,
            "VarRows",
            docstring=(
                "variance over the last dim (rows, cols) -> (rows,), divided by"
                " (cols - correction), float32 accumulation"
            ),
        )
        b.def_py_c_function(
            _log_softmax_rows_dispatcher,
            "LogSoftmaxRows",
            docstring=(
                "log-softmax over the last dim (rows, cols) -> (rows, cols)"
            ),
        )
        b.def_py_c_function(
            _anyall_rows_dispatcher[False],
            "AnyRows",
            docstring=(
                "any (nonzero) over the last dim (rows, cols) -> bool (rows,)"
            ),
        )
        b.def_py_c_function(
            _anyall_rows_dispatcher[True],
            "AllRows",
            docstring=(
                "all (nonzero) over the last dim (rows, cols) -> bool (rows,)"
            ),
        )
        return b.finalize()
    except e:
        abort(t"failed to create reduction_ops python module: {e}")
