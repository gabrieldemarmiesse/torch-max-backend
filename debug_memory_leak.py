import os
import shutil
from pathlib import Path

import numpy as np
import psutil
from max import dtype as max_dtype
from max.driver import CPU
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType
from mojo.paths import _build_mojo_source_package

max_cache = Path.home() / ".local/share/modular/.max_cache"
if max_cache.exists():
    print("Removing MAX cache directory at", max_cache)
    shutil.rmtree(max_cache)

mojo_cache = Path.home() / ".local/share/modular/.mojo_cache"
if mojo_cache.exists():
    print("Removing Mojo cache directory at", mojo_cache)
    shutil.rmtree(mojo_cache)

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


def _current_rss_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


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
    previous_rss = None

    print("i | rss_mb | delta_mb")

    for i in range(10_000, 10_005):
        print(i)
        np.array(run_add_one(custom_extension, i))
        rss = _current_rss_mb()
        delta = "-" if previous_rss is None else f"{rss - previous_rss:+.2f}"
        print(f"  memory: {rss:.2f} MB (delta {delta} MB)")
        previous_rss = rss
