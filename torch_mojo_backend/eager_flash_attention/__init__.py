"""Lazy loader for the separately cached FA4 Mojo extension.

The large vendored kernels deliberately live outside ``eager_kernels`` so a
FlashAttention change does not invalidate every ordinary eager extension.
"""

import fcntl
import hashlib
import importlib
from pathlib import Path

import mojo.importer  # noqa: F401 - installs the .mojo import hook

_PACKAGE_DIR = Path(__file__).parent
_CACHE_DIR = _PACKAGE_DIR / "__mojocache__"
_MODULE = None


def _sources_hash() -> str:
    hasher = hashlib.sha256()
    for path in sorted(_PACKAGE_DIR.rglob("*.mojo")):
        hasher.update(str(path.relative_to(_PACKAGE_DIR)).encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


def load_fa4_ops():
    """Compile once on first eligible use, then return the bridge module."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    cache_file = _CACHE_DIR / f"fa4_ops.hash-{_sources_hash()}.so"
    if cache_file.is_file():
        _MODULE = importlib.import_module(
            "torch_mojo_backend.eager_flash_attention.fa4_ops"
        )
        return _MODULE

    _CACHE_DIR.mkdir(exist_ok=True)
    with open(_CACHE_DIR / ".compile.lock", "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        _MODULE = importlib.import_module(
            "torch_mojo_backend.eager_flash_attention.fa4_ops"
        )
    return _MODULE


__all__ = ["load_fa4_ops"]
