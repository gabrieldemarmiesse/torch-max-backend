import os

if os.environ.get("TORCH_MAX_BACKEND_BEARTYPE", "1") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()


from torch_max_backend.custom_torch_ops_in_mojo.torch_custom_ops import (
    make_torch_op_from_mojo,
)
from torch_max_backend.max_device.log_aten_calls import log_aten_calls
from torch_max_backend.max_device.register import register_max_devices
from torch_max_backend.max_device.torch_max_tensor import TorchMaxTensor
from torch_max_backend.torch_compile_backend.compiler import (
    MAPPING_TORCH_ATEN_TO_MAX,
    MaxCompilerError,
    get_accelerators,
    max_backend,
)

__all__ = [
    "max_backend",
    "get_accelerators",
    "MAPPING_TORCH_ATEN_TO_MAX",
    "MaxCompilerError",
    "register_max_devices",
    "make_torch_op_from_mojo",
    "TorchMaxTensor",
    "log_aten_calls",
]
