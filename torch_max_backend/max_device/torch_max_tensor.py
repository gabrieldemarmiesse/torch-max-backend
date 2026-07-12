import functools
import math
from typing import no_type_check

import max.driver
import torch
from max.driver import CPU
from max.dtype import DType
from max.experimental.torch import max_dtype_to_torch

from torch_max_backend.max_device import torch_max_device_module

# The Mojo extension module (torch_max_backend.eager_kernels.tensor_holder),
# resolved lazily so that importing torch_max_backend never triggers a Mojo
# kernel compile.
_tensor_holder = None


def _holder_mod():
    global _tensor_holder
    if _tensor_holder is None:
        from torch_max_backend import eager_kernels

        _tensor_holder = eager_kernels.tensor_holder
    return _tensor_holder


_data_movement = None


def _data_movement_mod():
    global _data_movement
    if _data_movement is None:
        from torch_max_backend import eager_kernels

        _data_movement = eager_kernels.data_movement_ops
    return _data_movement


@no_type_check
def _ctx_ptr(device):
    # Rebinds this module-level name to the real (cached) implementation on
    # first use, so the lazy import costs one call, not one per call.
    global _ctx_ptr
    from torch_max_backend.eager_kernels import _ctx_ptr as real_ctx_ptr

    _ctx_ptr = real_ctx_ptr
    return real_ctx_ptr(device)


# The helpers below run several times per op dispatch (hundreds of times per
# transformer decode step); @no_type_check keeps beartype's import hook from
# adding a wrapper frame to each call.


@no_type_check
def _row_major_strides(shape) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


@no_type_check
def _compute_contiguous(shape, strides) -> bool:
    """torch's relaxed contiguity: size-1 dims never break contiguity."""
    expected = 1
    for size, stride in zip(reversed(shape), reversed(strides)):
        if size == 1:
            continue
        if stride != expected:
            return False
        expected *= size
    return True


# max DType -> torch dtype, cached as a plain dict: max_dtype_to_torch is
# called once per tensor wrapper created (~600/decode step).
_TORCH_DTYPE_OF: dict = {}


@no_type_check
def _torch_dtype_of(dtype):
    td = _TORCH_DTYPE_OF.get(dtype)
    if td is None:
        td = _TORCH_DTYPE_OF[dtype] = max_dtype_to_torch(dtype)
    return td


# Strided kernels take shapes/strides padded to rank 8 with LEADING entries.
MAX_RANK = 8


@no_type_check
def _pad8(values, fill: int) -> tuple[int, ...]:
    values = tuple(values)
    if len(values) > MAX_RANK:
        raise NotImplementedError(
            f"mojo tensors support at most rank {MAX_RANK}, got {len(values)}"
        )
    return (fill,) * (MAX_RANK - len(values)) + values


class TorchMojoTensor(torch.Tensor):
    """Eager mojo tensor.

    A meta-backed `PrivateUse1` wrapper (`torch._C._acc.create_empty_tensor`
    + `__class__` swap) whose payload is:

    - `_holder`: a Mojo `TensorHolder` owning the device allocation. Views
      share the *same* holder object; CPython's refcount on it is the
      ownership mechanism, and the last drop enqueues the stream-ordered
      free (see docs/strided_owning_tensors_design.md).
    - Layout metadata as plain Python attributes (`_ptr`, `_shape`,
      `_strides` in elements, `_offset` in elements from the allocation
      start, `_dtype` as a max DType, `_numel`, `_itemsize`, `_device`,
      `_is_contiguous`).

    PyTorch's own TensorImpl always reports contiguous strides; that is fine
    because the backend registers a kernel for every op that consumes a
    mojo tensor, so the real strides here are the only ones ever used.
    """

    @classmethod
    @no_type_check
    def _make(
        cls, holder, ptr, shape, strides, offset, dtype, device, contiguous=None
    ) -> "TorchMojoTensor":
        shape = tuple(shape)
        strides = tuple(strides)
        res = torch._C._acc.create_empty_tensor(shape, _torch_dtype_of(dtype))
        res.__class__ = TorchMojoTensor
        res._holder = holder
        res._ptr = ptr
        res._shape = shape
        res._strides = strides
        res._offset = offset
        res._dtype = dtype
        res._itemsize = dtype.size_in_bytes
        res._numel = math.prod(shape)
        res._device = device
        res._is_contiguous = (
            _compute_contiguous(shape, strides) if contiguous is None else contiguous
        )
        return res

    @classmethod
    @no_type_check
    def _alloc(
        cls, shape, dtype: DType, device: max.driver.Device
    ) -> "TorchMojoTensor":
        """A new contiguous uninitialized tensor (one device allocation)."""
        shape = tuple(shape)
        numel = math.prod(shape)
        holder, ptr = _holder_mod().alloc(_ctx_ptr(device), numel * dtype.size_in_bytes)
        return cls._make(
            holder,
            ptr,
            shape,
            _row_major_strides(shape),
            0,
            dtype,
            device,
            contiguous=True,
        )

    @classmethod
    @no_type_check
    def _view_of(
        cls, base: "TorchMojoTensor", shape, strides, offset, contiguous=None
    ) -> "TorchMojoTensor":
        """A zero-copy view: shares base's holder, new layout metadata.

        `offset` is absolute, in elements from the allocation start.
        `contiguous` skips the contiguity rescan when the caller knows it.
        """
        ptr = base._ptr + (offset - base._offset) * base._itemsize
        return cls._make(
            base._holder,
            ptr,
            shape,
            strides,
            offset,
            base._dtype,
            base._device,
            contiguous=contiguous,
        )

    @classmethod
    @no_type_check
    def _from_cpu(
        cls, cpu_tensor: torch.Tensor, device: max.driver.Device
    ) -> "TorchMojoTensor":
        """H2D: allocate + copy from a CPU torch tensor (synchronizes)."""
        from max.experimental.torch.torch import torch_dtype_to_max

        t = cpu_tensor.detach()
        if not t.is_contiguous():
            t = t.contiguous()
        dtype = torch_dtype_to_max(t.dtype)
        nbytes = t.numel() * t.element_size()
        if nbytes == 0:
            # Nothing to transfer; skip alloc_from_host's full queue drain.
            return cls._alloc(tuple(t.shape), dtype, device)
        holder, ptr = _holder_mod().alloc_from_host(
            _ctx_ptr(device), t.data_ptr(), nbytes
        )
        return cls._make(
            holder,
            ptr,
            tuple(t.shape),
            _row_major_strides(t.shape),
            0,
            dtype,
            device,
            contiguous=True,
        )

    @no_type_check
    def _to_cpu_tensor(self) -> torch.Tensor:
        """D2H: a CPU torch tensor with this tensor's data (synchronizes)."""
        src = self if self._is_contiguous else self._materialize_contiguous()
        out = torch.empty(self._shape, dtype=max_dtype_to_torch(self._dtype))
        if src._numel > 0:
            _holder_mod().copy_to_host(
                _ctx_ptr(src._device),
                src._ptr,
                out.data_ptr(),
                src._numel * src._itemsize,
            )
        return out

    @no_type_check
    def _materialize_contiguous(self) -> "TorchMojoTensor":
        """A new contiguous tensor with this tensor's (strided) contents."""
        out = TorchMojoTensor._alloc(self._shape, self._dtype, self._device)
        if self._numel > 0:
            rank = len(self._shape)
            if rank <= 4:
                # Hot path (attention q/k/v transposes, expand): the rank-4
                # PermuteCopy gathers a strided source into a contiguous
                # destination with no destination index math and half the
                # coordinate math of the generic rank-8 CopyStrided.
                pad = 4 - rank
                dims4 = (1,) * pad + tuple(self._shape)
                strides4 = (0,) * pad + tuple(self._strides)
                _data_movement_mod().PermuteCopy(
                    out._ptr,
                    self._ptr,
                    dims4,
                    strides4,
                    self._itemsize,
                    _ctx_ptr(self._device),
                )
            else:
                _copy_strided_into(out, self)
        return out

    @no_type_check
    def _contig(self) -> "TorchMojoTensor":
        """self if already contiguous, else a materialized copy."""
        return self if self._is_contiguous else self._materialize_contiguous()

    @no_type_check
    def __dlpack__(self, *, stream=None, **_unused):
        """Export the device allocation as a "dltensor" capsule.

        torch's inherited `__dlpack__` would export the zero-byte meta
        storage; this override exports the real allocation described by the
        Python-side metadata. Non-contiguous tensors are materialized first,
        and the capsule pins the (materialized) tensor's holder. `stream` is
        ignored: producers and consumers share the device's default stream
        (the same assumption the eager kernels make).
        """
        from torch_max_backend.max_device import dlpack

        src = self._contig()
        return dlpack.make_capsule(
            src._holder, src._ptr, src._shape, src._dtype, src._device
        )

    @no_type_check
    def __dlpack_device__(self):
        from torch_max_backend.max_device import dlpack

        return dlpack.dlpack_device(self._device)

    @no_type_check
    def __coerce_same_metadata_as_tangent__(self, expected_meta, expected_type=None):
        """Accept mojo tensors as backward tangents under torch.compile.

        AOTAutograd guesses tangent types from fake tensors, which are plain
        `torch.Tensor`s, and rejects runtime tangents of unexpected types
        unless this hook coerces them. A mojo tensor behaves exactly like a
        plain tensor for dispatch purposes, so no conversion is needed.
        """
        if expected_type not in (None, torch.Tensor):
            return None
        return self

    def __repr__(self):
        if hasattr(self, "_holder"):
            return f"TorchMojoTensor({self._to_cpu_tensor()!r}, device='{self.device}')"
        return super().__repr__()

    @property
    def device(self):
        if hasattr(self, "_device"):
            if self._device == CPU():
                return torch_max_device_module.cpu()
            return torch.device(f"mojo:{self._device.id}")
        return super().device

    __torch_function__ = torch._C._disabled_torch_function_impl


@no_type_check
def _rebind_payload(dst: TorchMojoTensor, src: TorchMojoTensor) -> None:
    """Point dst's payload at src's data — the out-variant "resize" pattern.

    torch's out= ops resize `out` when the shape doesn't match; our meta
    wrapper's torch-side shape is frozen, but nothing ever reads it — all
    consumers use the Python-side metadata rebound here (this mirrors the
    old `out._max_data = result` behavior).
    """
    dst._holder = src._holder
    dst._ptr = src._ptr
    dst._shape = src._shape
    dst._strides = src._strides
    dst._offset = src._offset
    dst._dtype = src._dtype
    dst._itemsize = src._itemsize
    dst._numel = src._numel
    dst._is_contiguous = src._is_contiguous
    # The cached Mojo TensorSpec describes the OLD payload; rebinding is the
    # one sanctioned metadata mutation, so swap (or drop) the spec here too.
    spec = src.__dict__.get("_spec")
    if spec is not None:
        dst._spec = spec
    else:
        dst.__dict__.pop("_spec", None)


@no_type_check
def _copy_strided_into(dst: TorchMojoTensor, src: TorchMojoTensor) -> None:
    """dst[coords] = src[coords]; same shape and dtype, any strides.

    The shared materialize/copy primitive: powers .contiguous(), copy_ into
    views, and expand materialization (src strides may contain 0s).
    """
    _holder_mod().CopyStrided(
        dst._ptr,
        src._ptr,
        _pad8(dst._shape, 1),
        _pad8(dst._strides, 0),
        _pad8(src._strides, 0),
        dst._itemsize,
        _ctx_ptr(dst._device),
    )


@functools.cache
def get_ordered_accelerators():
    """Get accelerators ordered with GPUs first, then CPU last"""
    from torch_max_backend.torch_compile_backend.compiler import get_accelerators

    accelerators = list(get_accelerators())

    # Separate GPU and CPU accelerators
    gpu_accelerators = [acc for acc in accelerators if acc.label == "gpu"]
    cpu_accelerators = [acc for acc in accelerators if acc.label == "cpu"]

    # Order: GPUs first, then CPU last
    return gpu_accelerators + cpu_accelerators


@functools.cache
@no_type_check
def find_equivalent_max_device(device: torch.device) -> max.driver.Device:
    """Find the equivalent MAX device for a given torch device

    Device mapping:
    - mojo:0 (or mojo) -> First GPU (or CPU if no GPUs)
    - mojo:1, mojo:2, ... -> Additional GPUs
    - mojo:<last_index> -> CPU device
    """
    ordered_accelerators = get_ordered_accelerators()

    if device.type == "mojo":
        # mojo with specific index
        if device.index is None:
            # Default to first accelerator (first GPU or CPU if no GPUs)
            return ordered_accelerators[0]
        else:
            if device.index < len(ordered_accelerators):
                return ordered_accelerators[device.index]
            else:
                raise ValueError(f"Invalid mojo index {device.index}")
    elif device.type == "cpu":
        # Find CPU accelerator (should be last in ordered list)
        for acc in reversed(ordered_accelerators):  # Check from the end
            if acc.label == "cpu":
                return acc
        # If no CPU found, return last accelerator as fallback
        return ordered_accelerators[-1]
    elif device.type in ("cuda", "hip"):
        # Find GPU accelerator (should be first in ordered list)
        # TODO: allow setting the default device index globally like with cuda
        gpu_index = device.index if device.index is not None else 0
        gpu_accelerators = [acc for acc in ordered_accelerators if acc.label == "gpu"]
        if gpu_index < len(gpu_accelerators):
            return gpu_accelerators[gpu_index]
        raise RuntimeError(f"GPU index {gpu_index} not available in MAX")
    else:
        raise NotImplementedError(f"Cannot convert {device.type} to MAX device")
