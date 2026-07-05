"""Fast eager-mode kernels for max_device, compiled as CPython extensions.

The `.mojo` modules in this package are imported through `mojo.importer`
(the official Mojo import hook): the first import compiles each module with
`mojo build --emit shared-lib` and caches the resulting extension under
`__mojocache__/` (content addressed — recompiles only when a `.mojo` file
in this directory changes).

This bypasses the `max.experimental.tensor` per-op pipeline (graph building
+ MLIR passes + interpreter/compiler) entirely: an op call here is one
CPython extension call that unwraps `max.driver.Buffer` pointers and
enqueues a kernel on MAX's own DeviceContext, so it stays correctly ordered
with every other MAX driver operation on that device.

All kernels use fully dynamic shapes: one compiled extension serves every
shape and dtype (dtype dispatch happens at runtime inside Mojo).
"""

import fcntl
import hashlib
import sys
from pathlib import Path

import mojo.importer  # noqa: F401  — installs the .mojo meta-path importer
from max import driver
from max.experimental.tensor import Tensor as MaxEagerTensor

_PACKAGE_DIR = Path(__file__).parent
_CACHE_DIR = _PACKAGE_DIR / "__mojocache__"


def _mojo_sources_hash() -> str:
    """Content hash of the package's .mojo sources.

    Must match `mojo.importer._calculate_mojo_source_hash` so we can
    predict the cache filename the importer will look for.
    """
    hasher = hashlib.sha256()
    for file_path in sorted(_PACKAGE_DIR.rglob("*.mojo")):
        hasher.update(str(file_path.relative_to(_PACKAGE_DIR)).encode())
        hasher.update(file_path.read_bytes())
    return hasher.hexdigest()[:16]


def _import_mojo_modules():
    """Import (compiling on first use) the .mojo extension modules.

    `mojo.importer` has no cross-process lock: concurrent cold-cache
    imports (e.g. pytest-xdist workers) would each run `mojo build`
    writing to the same cache file. Serialize the compile with a file
    lock; once the cache file for the current source hash exists, the
    import is a plain dlopen and needs no lock.
    """
    cache_file = _CACHE_DIR / f"elementwise_ops.hash-{_mojo_sources_hash()}.so"
    if cache_file.is_file():
        from torch_max_backend.eager_kernels import elementwise_ops

        return elementwise_ops

    _CACHE_DIR.mkdir(exist_ok=True)
    with open(_CACHE_DIR / ".compile.lock", "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if not cache_file.is_file():
            print(
                "torch-max-backend: compiling eager-mode Mojo kernels "
                "(first use only, takes ~30s, cached afterwards)...",
                file=sys.stderr,
            )
        from torch_max_backend.eager_kernels import elementwise_ops

        return elementwise_ops


elementwise_ops = _import_mojo_modules()

_CTX_PTR_CACHE: dict[driver.Device, int] = {}


class FastPathUnavailable(Exception):
    """Raised when inputs don't qualify for the fast kernel path."""


def _ctx_ptr(device: driver.Device) -> int:
    ptr = _CTX_PTR_CACHE.get(device)
    if ptr is None:
        ptr = device._device_context_ptr()
        _CTX_PTR_CACHE[device] = ptr
    return ptr


def _driver_buffer(tensor: MaxEagerTensor) -> driver.Buffer:
    if not tensor.real:
        raise FastPathUnavailable("tensor is not realized")
    buffer = tensor.driver_tensor
    if not buffer.is_contiguous:
        raise FastPathUnavailable("tensor is not contiguous")
    return buffer


def binary_op(mojo_fn, lhs: MaxEagerTensor, rhs: MaxEagerTensor) -> MaxEagerTensor:
    """Run an elementwise binary kernel on two identically-shaped tensors."""
    lhs_buffer = _driver_buffer(lhs)
    rhs_buffer = _driver_buffer(rhs)
    if lhs_buffer.dtype != rhs_buffer.dtype:
        raise FastPathUnavailable("mismatched dtypes")
    if lhs_buffer.shape != rhs_buffer.shape:
        raise FastPathUnavailable("mismatched shapes (broadcasting)")
    if lhs_buffer.device != rhs_buffer.device:
        raise FastPathUnavailable("mismatched devices")
    out = driver.Buffer(lhs_buffer.dtype, lhs_buffer.shape, lhs_buffer.device)
    # Zero-sized buffers: nothing to compute, and empty buffers can have
    # sentinel data pointers that don't survive the pointer round-trip.
    if lhs_buffer.num_elements > 0:
        mojo_fn(out, lhs_buffer, rhs_buffer, _ctx_ptr(lhs_buffer.device))
    return MaxEagerTensor(storage=out)


def unary_op(mojo_fn, x: MaxEagerTensor) -> MaxEagerTensor:
    """Run an elementwise unary kernel."""
    x_buffer = _driver_buffer(x)
    out = driver.Buffer(x_buffer.dtype, x_buffer.shape, x_buffer.device)
    if x_buffer.num_elements > 0:
        mojo_fn(out, x_buffer, _ctx_ptr(x_buffer.device))
    return MaxEagerTensor(storage=out)
