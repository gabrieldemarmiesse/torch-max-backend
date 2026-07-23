"""Eager multi-GPU collectives for the mojo device.

Single-process collectives built on Modular's pure-Mojo P2P comm kernels
(M1 of docs/multi_gpu_training_plan.md): one Python call takes one tensor
per GPU and enqueues the collective on every device's default stream. No
NCCL, no torch-cuda — works with a CPU-only PyTorch install.

    from torch_mojo_backend import distributed as mojo_dist

    grads = [g0, g1]          # one tensor per GPU, same shape/dtype
    mojo_dist.all_reduce(grads, op="mean")
"""

import threading
from collections.abc import Callable

import torch
from max.driver import Device as MaxDevice
from max.dtype import DType

from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor

_COMM_TORCH_DTYPES = (torch.float32, torch.bfloat16, torch.float16)

_MIN_PAYLOAD_BYTES = 32 * 1024 * 1024


class _Signals:
    """Zeroed per-device Signal buffers for one ordered device set."""

    def __init__(self, devices: tuple[MaxDevice, ...], payload_bytes: int):
        from torch_mojo_backend import eager_kernels

        header = int(eager_kernels.comm_ops.signal_header_bytes())
        self.payload_bytes = payload_bytes
        nbytes = header + len(devices) * payload_bytes
        self.buffers = [
            TorchMojoTensor._alloc((nbytes,), DType.uint8, device) for device in devices
        ]
        for buffer in self.buffers:
            torch.ops.aten.zero_(buffer)
        # The barrier protocol requires every rank's counters to be zero
        # before ANY rank's first collective kernel can run; a peer would
        # otherwise read garbage through P2P. One synchronize per device at
        # allocation time guarantees it.
        for buffer in self.buffers:
            buffer._device.default_stream.synchronize()


# Per-device Signal buffers, allocated once per (devices, payload capacity)
# and reused by every collective: the comm kernels' cross-GPU barrier
# counters are monotonic, so consecutive collectives need no reinitialization.
_signal_cache: dict[tuple[MaxDevice, ...], _Signals] = {}
_signal_lock = threading.Lock()


def _signals_for(devices: tuple[MaxDevice, ...], payload_bytes: int) -> _Signals:
    with _signal_lock:
        signals = _signal_cache.get(devices)
        if signals is None or signals.payload_bytes < payload_bytes:
            capacity = max(_MIN_PAYLOAD_BYTES, payload_bytes)
            if signals is not None:
                capacity = max(capacity, 2 * signals.payload_bytes)
            signals = _Signals(devices, capacity)
            _signal_cache[devices] = signals
        return signals


def _check_collective_inputs(tensors: list[torch.Tensor]) -> None:
    if len(tensors) < 2:
        raise ValueError(f"need one tensor per GPU (>= 2), got {len(tensors)}")
    first = tensors[0]
    for i, tensor in enumerate(tensors):
        if not isinstance(tensor, TorchMojoTensor):
            raise ValueError(f"tensors[{i}] is not a mojo tensor: {type(tensor)}")
        if tensor._device.label != "gpu":
            raise ValueError(f"tensors[{i}] is not on a GPU: {tensor.device}")
        # The comm kernels derive each instance's rank from the device id
        # (ctx.id()) and index peer pointers with it, so tensors[i] must
        # live on the GPU whose id is exactly i.
        if tensor._device.id != i:
            raise ValueError(
                f"tensors[{i}] must be on mojo:{i} (device ids must be "
                f"0..{len(tensors) - 1} in order), got {tensor.device}"
            )
        if tuple(tensor._shape) != tuple(first._shape):
            raise ValueError(
                f"tensors[{i}] shape {tuple(tensor._shape)} != "
                f"tensors[0] shape {tuple(first._shape)}"
            )
        if tensor._dtype != first._dtype:
            raise ValueError(f"tensors[{i}] dtype {tensor._dtype} != {first._dtype}")
    if first.dtype not in _COMM_TORCH_DTYPES:
        raise ValueError(
            f"unsupported collective dtype {first.dtype}; "
            f"supported: {_COMM_TORCH_DTYPES}"
        )


# Collective launches are serialized: concurrent launches could enqueue in
# opposite per-device stream orders (each device's enqueue runs on its own
# AsyncRT worker) and deadlock the cross-GPU barrier, and a signal-buffer
# grow must not run while another launch is mid-enqueue. Holding the lock
# until every instance is enqueued (the Mojo call returns after its
# TaskGroup wait) makes per-stream order identical everywhere, and the
# device synchronizes inside _Signals.__init__ then cover all in-flight
# users of the buffers being replaced.
_launch_lock = threading.Lock()

# The comm kernels vector-load at align_of[SIMD[dtype, simd_width]] from the
# base pointer; fresh allocations satisfy this, arbitrary view offsets don't.
_COMM_PTR_ALIGN = 16


def _aligned_contig(tensor: TorchMojoTensor) -> TorchMojoTensor:
    src = tensor._contig()
    if src._ptr % _COMM_PTR_ALIGN:
        src = src._materialize_contiguous()
    return src


def _all_reduce_into(
    outs: list[TorchMojoTensor], srcs: list[TorchMojoTensor], average: bool
) -> None:
    """Enqueue out[i] = reduce(srcs) on every device. No host sync."""
    from torch_mojo_backend import eager_kernels

    devices = tuple(t._device for t in srcs)
    numel = srcs[0]._numel
    nbytes = numel * srcs[0]._itemsize
    with _launch_lock:
        signals = _signals_for(devices, nbytes)
        eager_kernels.comm_ops.all_reduce(
            tuple(t._ptr for t in srcs),
            tuple(t._ptr for t in outs),
            tuple(s._ptr for s in signals.buffers),
            tuple(eager_kernels._ctx_ptr(device) for device in devices),
            numel,
            srcs[0]._dtype.value,
            len(srcs),
            average,
        )


def all_reduce(tensors: list[torch.Tensor], op: str = "sum") -> None:
    """Allreduce in place: every tensor ends up holding reduce(tensors).

    Args:
        tensors: One tensor per participating GPU (distinct devices), all
            with identical shape and dtype (float32, bfloat16 or float16).
        op: "sum" or "mean" ("avg" is accepted as an alias).
    """
    if op not in ("sum", "mean", "avg"):
        raise ValueError(f"unsupported reduce op: {op!r}")
    _check_collective_inputs(tensors)
    if tensors[0]._numel == 0:
        return

    srcs = [_aligned_contig(t) for t in tensors]
    # The latency-bound kernel writes each output while peers still read
    # the inputs, so reduce out of place, then copy back on-device.
    outs = [TorchMojoTensor._alloc(src._shape, src._dtype, src._device) for src in srcs]
    _all_reduce_into(outs, srcs, average=op in ("mean", "avg"))
    for tensor, out in zip(tensors, outs):
        torch.ops.aten.copy_(tensor, out)


def all_reduce_out(tensors: list[torch.Tensor], op: str = "sum") -> list[torch.Tensor]:
    """Allreduce returning fresh output tensors (inputs left untouched)."""
    if op not in ("sum", "mean", "avg"):
        raise ValueError(f"unsupported reduce op: {op!r}")
    _check_collective_inputs(tensors)
    srcs = [_aligned_contig(t) for t in tensors]
    outs = [TorchMojoTensor._alloc(src._shape, src._dtype, src._device) for src in srcs]
    if srcs[0]._numel:
        _all_reduce_into(outs, srcs, average=op in ("mean", "avg"))
    return outs


# ---------------------------------------------------------------------------
# torch.distributed integration: a pure-Python ProcessGroup with
# thread-per-rank ranks in ONE process (M2 of docs/multi_gpu_training_plan.md).
# Rank threads rendezvous per collective; the last-arriving rank launches the
# single multi-device operation for everyone. Works with standard
# DistributedDataParallel via ``DDP(model, process_group=pg)``.
# ---------------------------------------------------------------------------


# Marks a rank slot that has not received its payload yet (payloads may
# legitimately be None, e.g. barrier).
_NOT_DEPOSITED = object()


class RankExitedError(RuntimeError):
    """A peer rank exited (crashed or returned) mid-collective."""


class _Collective:
    """One in-flight collective: deposit per-rank payloads, run once."""

    def __init__(self, world_size: int):
        self.condition = threading.Condition()
        self.world_size = world_size
        self.slots: list[object] = [_NOT_DEPOSITED] * world_size
        self.arrived = 0
        self.done = False
        self.executed_by: int | None = None
        self.exception: BaseException | None = None

    def join(
        self, rank: int, payload: object, fn: Callable[[list[object]], None]
    ) -> None:
        """Deposit ``payload``; the last arrival executes ``fn(slots)``."""
        with self.condition:
            self.slots[rank] = payload
            self.arrived += 1
            if self.arrived == self.world_size and not self.done:
                self.executed_by = rank
                try:
                    fn(self.slots)
                except BaseException as exc:
                    self.exception = exc
                self.done = True
                self.condition.notify_all()
            else:
                self.condition.wait_for(lambda: self.done)
        if self.exception is not None:
            if self.executed_by == rank:
                raise self.exception
            # Re-raising one exception object on several threads races on
            # its __traceback__; waiters raise their own wrapper instead.
            if isinstance(self.exception, RankExitedError):
                raise RankExitedError(str(self.exception)) from self.exception
            raise RuntimeError(
                f"collective failed on rank {self.executed_by}: {self.exception!r}"
            ) from self.exception

    def abort(self, exception: BaseException) -> None:
        """Fail the collective and wake every waiting rank."""
        with self.condition:
            if self.done:
                return
            self.exception = exception
            self.done = True
            self.condition.notify_all()


class _Comm:
    """Rendezvous shared by all rank threads of one logical process group.

    Ranks issue the same collective sequence (SPMD), so a per-rank monotonic
    op counter matches concurrent collectives across threads even when one
    rank races ahead into the next collective. A rank that exits — crashed
    or returned — while peers are (or later go) inside a collective poisons
    the rendezvous instead of leaving them parked forever.
    """

    def __init__(self, world_size: int):
        self.world_size = world_size
        self._lock = threading.Lock()
        self._pending: dict[int, _Collective] = {}
        self._exited_ranks: set[int] = set()

    def collective(
        self, op_id: int, rank: int, payload: object, fn: Callable[[list[object]], None]
    ) -> None:
        with self._lock:
            if self._exited_ranks:
                raise RankExitedError(
                    f"rank(s) {sorted(self._exited_ranks)} already exited; "
                    "this collective can never complete"
                )
            coll = self._pending.get(op_id)
            if coll is None:
                coll = self._pending[op_id] = _Collective(self.world_size)
                # The entry only matches arrivals for this op_id; the ids
                # grow monotonically so it can be dropped eagerly once full.
        try:
            coll.join(rank, payload, fn)
        finally:
            with self._lock:
                if coll.arrived == self.world_size:
                    self._pending.pop(op_id, None)

    def rank_exited(self, rank: int) -> None:
        """Record that ``rank``'s thread is gone and fail what waits on it.

        Collectives the rank already contributed to are complete or
        completing; only the ones still missing its deposit can never
        finish, so those (and all future collectives) fail fast.
        """
        with self._lock:
            self._exited_ranks.add(rank)
            pending = [
                coll
                for coll in self._pending.values()
                if coll.slots[rank] is _NOT_DEPOSITED
            ]
        for coll in pending:
            coll.abort(
                RankExitedError(
                    f"rank {rank} exited while this collective was waiting for it"
                )
            )


def _reduce_via_host(gathered: list[torch.Tensor], average: bool) -> None:
    """Correctness fallback for dtypes the comm kernels do not cover."""
    total = sum(t.cpu().to(torch.float64) for t in gathered)
    if average:
        total = total / len(gathered)
    for tensor in gathered:
        tensor.copy_(total.to(tensor.dtype))


def _reduce_tensors(gathered: list[torch.Tensor], average: bool) -> None:
    """In-place allreduce of one tensor per rank (any devices/dtype)."""
    first = gathered[0]
    same_shape = all(tuple(t.shape) == tuple(first.shape) for t in gathered)
    if not same_shape:
        raise ValueError(
            f"allreduce shape mismatch across ranks: "
            f"{[tuple(t.shape) for t in gathered]}"
        )
    fast = (
        first.dtype in _COMM_TORCH_DTYPES
        and all(isinstance(t, TorchMojoTensor) for t in gathered)
        and all(t._device.label == "gpu" for t in gathered)
        # The comm kernels index peers by device id; anything else takes
        # the host-staged path.
        and all(t._device.id == i for i, t in enumerate(gathered))
    )
    if fast:
        all_reduce(gathered, op="mean" if average else "sum")
    else:
        _reduce_via_host(gathered, average)


def _make_work(result: object) -> torch.distributed.Work:
    """A completed c10d Work wrapping ``result`` (a list of tensors)."""
    from torch._C._distributed_c10d import _create_work_from_future
    from torch.futures import Future

    future = Future()
    future.set_result(result)
    return _create_work_from_future(future)


class MojoProcessGroup(torch.distributed.ProcessGroup):
    """A synchronous c10d process group over thread-per-rank mojo ranks.

    One instance per rank thread, all sharing one :class:`_Comm`. Every
    collective rendezvouses the rank threads; the last arrival executes the
    whole multi-device operation (allreduce on the P2P comm kernels,
    broadcast/allgather with direct D2D copies), then all ranks return
    already-completed Work objects — the synchronous process group contract
    that DistributedDataParallel supports out of the box.

    Restrictions: ``find_unused_parameters=True`` and ``static_graph=True``
    are unsupported (they need a pinned-memory allocator this backend does
    not register).
    """

    def __init__(self, rank: int, world_size: int, comm: _Comm):
        super().__init__(rank, world_size)
        if comm.world_size != world_size:
            raise ValueError(
                f"comm world_size {comm.world_size} != group size {world_size}"
            )
        self._comm = comm
        self._rank_index = rank
        self._op_counter = 0

    def getBackendName(self) -> str:
        return "mojo"

    def _next_op_id(self) -> int:
        self._op_counter += 1
        return self._op_counter

    def _run(self, payload: object, fn: Callable[[list[object]], None]) -> None:
        self._comm.collective(self._next_op_id(), self._rank_index, payload, fn)

    # -- collectives ------------------------------------------------------

    def allreduce(
        self, tensors: list[torch.Tensor], opts: object = None
    ) -> torch.distributed.Work:
        reduce_op = torch.distributed.ReduceOp.SUM if opts is None else opts.reduceOp
        # Compare with reduce_op on the left: c10d ReduceOp instances
        # implement __eq__ against the RedOpType constants, not the reverse.
        is_sum = reduce_op == torch.distributed.ReduceOp.SUM
        average = reduce_op == torch.distributed.ReduceOp.AVG
        if not (is_sum or average):
            raise NotImplementedError(
                f"MojoProcessGroup.allreduce only supports SUM and AVG, got {reduce_op}"
            )

        def run(slots: list[object]) -> None:
            for position in range(len(tensors)):
                _reduce_tensors([s[position] for s in slots], average)

        self._run(tensors, run)
        return _make_work(tensors)

    def allreduce_coalesced(
        self, tensors: list[torch.Tensor], opts: object = None
    ) -> torch.distributed.Work:
        return self.allreduce(tensors, opts)

    def broadcast(
        self, tensors: list[torch.Tensor], opts: object = None
    ) -> torch.distributed.Work:
        root_rank = 0 if opts is None else opts.rootRank

        def run(slots: list[object]) -> None:
            root_tensors = slots[root_rank]
            for rank, rank_tensors in enumerate(slots):
                if rank == root_rank:
                    continue
                for source, tensor in zip(root_tensors, rank_tensors):
                    tensor.copy_(source)

        self._run(tensors, run)
        return _make_work(tensors)

    def allgather(
        self,
        output_tensors: list[list[torch.Tensor]],
        input_tensors: list[torch.Tensor],
        opts: object = None,
    ) -> torch.distributed.Work:
        def run(slots: list[object]) -> None:
            for _, outputs in slots:
                for list_index, per_rank_outputs in enumerate(outputs):
                    for source_rank, (source_inputs, _) in enumerate(slots):
                        per_rank_outputs[source_rank].copy_(source_inputs[list_index])

        self._run((input_tensors, output_tensors), run)
        return _make_work(output_tensors)

    def _allgather_base(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        opts: object = None,
    ) -> torch.distributed.Work:
        def run(slots: list[object]) -> None:
            for _, output in slots:
                chunks = output.chunk(self._comm.world_size)
                for source_rank, (source_input, _) in enumerate(slots):
                    chunks[source_rank].copy_(source_input)

        self._run((input_tensor, output_tensor), run)
        return _make_work([output_tensor])

    def barrier(self, opts: object = None) -> torch.distributed.Work:
        def run(slots: list[object]) -> None:
            pass

        self._run(None, run)
        return _make_work([])


def spawn(
    fn: Callable[[int, int, MojoProcessGroup], None], world_size: int | None = None
) -> None:
    """Run ``fn(rank, world_size, process_group)`` in one thread per rank.

    Thread-per-rank data parallelism in a single process: rank ``i`` is
    pinned to ``mojo:i`` (thread-local current device) with calling-thread
    autograd, and all ranks share one rendezvous. Standard DDP usage::

        def worker(rank, world_size, pg):
            model = DDP(model.to(f"mojo:{rank}"), process_group=pg)
            ...

        spawn(worker, world_size=2)

    The first rank's exception (if any) is re-raised after all threads stop.
    """
    from torch_mojo_backend import register_mojo_devices
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
        get_ordered_accelerators,
    )

    register_mojo_devices()
    gpu_count = sum(acc.label == "gpu" for acc in get_ordered_accelerators())
    if world_size is None:
        world_size = gpu_count
    if world_size < 2:
        raise ValueError(f"spawn needs world_size >= 2, got {world_size}")
    if world_size > gpu_count:
        raise ValueError(
            f"world_size {world_size} exceeds available GPUs ({gpu_count})"
        )

    comm = _Comm(world_size)
    errors: list[BaseException | None] = [None] * world_size

    def runner(rank: int) -> None:
        torch.mojo.set_device(rank)
        # The autograd engine must run on this thread: the Python
        # PrivateUse1 guard advertises a single engine device queue, and
        # concurrent per-rank backward passes need per-thread execution.
        torch.autograd.set_multithreading_enabled(False)
        process_group = MojoProcessGroup(rank, world_size, comm)
        try:
            fn(rank, world_size, process_group)
        except BaseException as exc:
            errors[rank] = exc
        finally:
            # Whether crashed or returned, this rank will never join another
            # collective; fail the ones waiting on it instead of hanging.
            comm.rank_exited(rank)

    threads = [
        threading.Thread(target=runner, args=(rank,), name=f"mojo-rank-{rank}")
        for rank in range(world_size)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    # Prefer the root cause: a rank that failed on its own, not one that
    # merely observed a peer's exit through a poisoned collective.
    real_errors = [
        e for e in errors if e is not None and not isinstance(e, RankExitedError)
    ]
    for error in real_errors or errors:
        if error is not None:
            raise error
