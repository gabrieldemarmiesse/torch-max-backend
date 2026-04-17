from pathlib import Path

import numpy as np
from max import dtype as max_dtype
from max.driver import CPU
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType
from mojo.paths import _build_mojo_source_package

custom_op_dir = Path("/tmp/kernels_eager")
custom_kernel_dir = custom_op_dir / "kernels"
custom_op_dir.mkdir(parents=True, exist_ok=True)
custom_kernel_dir.mkdir(parents=True, exist_ok=True)
(custom_kernel_dir / "__init__.mojo").touch()

kernel_source = """
import compiler
from tensor import ElementwiseUnaryOp

@compiler.register("add_one")
struct AddOne(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: Int,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return x + 1
"""

(custom_kernel_dir / "add_one.mojo").write_text(kernel_source)


def run_add_one(extension_path: Path, i):
    x = Tensor([list(range(i))], dtype=max_dtype.DType.int32, device=CPU())
    out_type = TensorType(dtype=x.dtype, shape=x.shape, device=x.device)

    y = F.custom(
        name="add_one",
        device=x.device,
        values=[x],
        out_types=[out_type],
        custom_extensions=[extension_path],
    )[0]

    return y


if __name__ == "__main__":
    custom_extension = _build_mojo_source_package(custom_kernel_dir)

    for i in range(1, 10):
        print(i)
        np.array(run_add_one(custom_extension, i))
