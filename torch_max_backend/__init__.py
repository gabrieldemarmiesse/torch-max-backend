import os

if os.environ.get("TORCH_MAX_BACKEND_BEARTYPE", "1") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()


from torch_max_backend.compiler import (
    max_backend,
    get_accelerators,
    MAPPING_TORCH_ATEN_TO_MAX,
)

__all__ = ["max_backend", "get_accelerators", "MAPPING_TORCH_ATEN_TO_MAX"]
