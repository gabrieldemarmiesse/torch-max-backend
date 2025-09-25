import os

if os.environ.get("TORCH_MAX_BACKEND_BEARTYPE", "1") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()


from torch_max_backend.compiler import (
    max_backend,
    get_accelerators,
    MAPPING_TORCH_ATEN_TO_MAX,
    MaxCompilerError,
)
from torch_max_backend.max_device import register_max_devices
from torch_max_backend.torch_custom_ops import make_torch_op_from_mojo

__all__ = [
    "max_backend",
    "get_accelerators",
    "MAPPING_TORCH_ATEN_TO_MAX",
    "MaxCompilerError",
    "register_max_devices",
    "make_torch_op_from_mojo",
]
