import torch
from torch.utils.backend_registration import _setup_privateuseone_for_python_backend

from .max_device_aten_ops import _aten_ops_registry

_registered = False


def register_max_devices():
    """Enable the max_device globally and register all aten ops"""
    global _registered
    if _registered:
        # Already registered
        return

    _setup_privateuseone_for_python_backend("max_device")

    # Register all collected aten operations
    for op_name, func in _aten_ops_registry:
        torch.library.impl(op_name, "privateuseone")(func)

    _registered = True
