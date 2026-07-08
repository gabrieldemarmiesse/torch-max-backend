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

Modules are grouped by category (elementwise, nn, data movement, matmul,
conv) and each is imported lazily on the first call of an op in that
category, so a given workload only compiles the categories it uses.
`matmul_ops` and `conv_ops` call into the MAX kernel library (`linalg`,
`nn`) — the same kernels the graph compiler uses, including the
cuBLAS/cuDNN vendor paths on NVIDIA GPUs.
"""

import fcntl
import hashlib
import importlib
import sys
from pathlib import Path
from typing import no_type_check

import mojo.importer  # noqa: F401  — installs the .mojo meta-path importer
from max import driver
from max.experimental.tensor import Tensor as MaxEagerTensor

_PACKAGE_DIR = Path(__file__).parent
_CACHE_DIR = _PACKAGE_DIR / "__mojocache__"

_MOJO_MODULES = (
    "tensor_holder",
    "elementwise_ops",
    "nn_ops",
    "data_movement_ops",
    "logic_ops",
    "matmul_ops",
    "conv_ops",
)


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


def _import_mojo_module(name: str):
    """Import (compiling on first use) one .mojo extension module.

    `mojo.importer` has no cross-process lock: concurrent cold-cache
    imports (e.g. pytest-xdist workers) would each run `mojo build`
    writing to the same cache file. Serialize the compile with a file
    lock; once the cache file for the current source hash exists, the
    import is a plain dlopen and needs no lock.
    """
    cache_file = _CACHE_DIR / f"{name}.hash-{_mojo_sources_hash()}.so"
    if cache_file.is_file():
        return importlib.import_module(f"torch_max_backend.eager_kernels.{name}")

    _CACHE_DIR.mkdir(exist_ok=True)
    with open(_CACHE_DIR / ".compile.lock", "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if not cache_file.is_file():
            print(
                f"torch-max-backend: compiling eager-mode Mojo kernels ({name}) "
                "(first use only, takes ~30s, cached afterwards)...",
                file=sys.stderr,
            )
        return importlib.import_module(f"torch_max_backend.eager_kernels.{name}")


def __getattr__(name: str):
    if name in _MOJO_MODULES:
        module = _import_mojo_module(name)
        globals()[name] = module  # later lookups skip __getattr__
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_CTX_PTR_CACHE: dict[driver.Device, int] = {}


class FastPathUnavailable(Exception):
    """Raised when inputs don't qualify for the fast kernel path."""


@no_type_check
def _ctx_ptr(device: driver.Device) -> int:
    ptr = _CTX_PTR_CACHE.get(device)
    if ptr is None:
        ptr = device._device_context_ptr()
        _CTX_PTR_CACHE[device] = ptr
    return ptr


@no_type_check
def _driver_buffer(tensor: MaxEagerTensor) -> driver.Buffer:
    if not tensor.real:
        raise FastPathUnavailable("tensor is not realized")
    buffer = tensor.driver_tensor
    if not buffer.is_contiguous:
        raise FastPathUnavailable("tensor is not contiguous")
    return buffer


@no_type_check
def binary_op(
    mojo_fn, lhs: MaxEagerTensor, rhs: MaxEagerTensor, out: driver.Buffer | None = None
) -> MaxEagerTensor:
    """Run an elementwise binary kernel on two identically-shaped tensors.

    Pass `out` to write into an existing buffer (in-place variants).
    """
    lhs_buffer = _driver_buffer(lhs)
    rhs_buffer = _driver_buffer(rhs)
    if lhs_buffer.dtype != rhs_buffer.dtype:
        raise FastPathUnavailable("mismatched dtypes")
    if lhs_buffer.shape != rhs_buffer.shape:
        raise FastPathUnavailable("mismatched shapes (broadcasting)")
    if lhs_buffer.device != rhs_buffer.device:
        raise FastPathUnavailable("mismatched devices")
    if out is None:
        out = driver.Buffer(lhs_buffer.dtype, lhs_buffer.shape, lhs_buffer.device)
    # Zero-sized buffers: nothing to compute, and empty buffers can have
    # sentinel data pointers that don't survive the pointer round-trip.
    if lhs_buffer.num_elements > 0:
        mojo_fn(out, lhs_buffer, rhs_buffer, _ctx_ptr(lhs_buffer.device))
    return MaxEagerTensor(storage=out)


@no_type_check
def unary_op(mojo_fn, x: MaxEagerTensor) -> MaxEagerTensor:
    """Run an elementwise unary kernel."""
    x_buffer = _driver_buffer(x)
    out = driver.Buffer(x_buffer.dtype, x_buffer.shape, x_buffer.device)
    if x_buffer.num_elements > 0:
        mojo_fn(out, x_buffer, _ctx_ptr(x_buffer.device))
    return MaxEagerTensor(storage=out)


@no_type_check
def scalar_op(mojo_fn, x: MaxEagerTensor, scalar: float) -> MaxEagerTensor:
    """Run an elementwise kernel with a Python-scalar second operand."""
    x_buffer = _driver_buffer(x)
    out = driver.Buffer(x_buffer.dtype, x_buffer.shape, x_buffer.device)
    if x_buffer.num_elements > 0:
        mojo_fn(out, x_buffer, float(scalar), _ctx_ptr(x_buffer.device))
    return MaxEagerTensor(storage=out)
