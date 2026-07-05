"""Benchmark the Mojo-extension fast eager path against the graph-based path.

Usage:
    uv run python scripts/benchmark_fast_eager.py

Compares, per op call on max_device:
  - the fast path (TORCH_MAX_BACKEND_FAST_EAGER=1, default)
  - the graph-based path (TORCH_MAX_BACKEND_FAST_EAGER=0)
by re-running itself in a subprocess for each mode, plus torch-native CPU
and CUDA references in-process.
"""

import os
import subprocess
import sys
import time

N_ITERS = 500
N_WARMUP = 50
SHAPES = [(64, 64), (1024, 1024)]


def bench(fn, label, iters=N_ITERS, warmup=N_WARMUP):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    print(f"    {label:44s} {(t1 - t0) / iters * 1e6:9.1f} us/call")


def run_max_device_benchmarks():
    import torch

    from torch_max_backend import register_max_devices
    from torch_max_backend.flags import fast_eager_enabled

    register_max_devices()
    mode = "fast (mojo extensions)" if fast_eager_enabled() else "graph-based"
    print(f"  max_device eager mode: {mode}")
    dev = torch.device("max_device")
    for shape in SHAPES:
        x = torch.randn(*shape).to(dev)
        y = torch.randn(*shape).to(dev)
        bench(lambda: x + y, f"add {shape}")
        bench(lambda: x * y, f"mul {shape}")
        bench(lambda: torch.relu(x), f"relu {shape}")


def run_torch_references():
    import torch

    print("  torch native (reference)")
    for shape in SHAPES:
        x = torch.randn(*shape)
        y = torch.randn(*shape)
        bench(lambda: x + y, f"cpu add {shape}")
        if torch.cuda.is_available():
            xg = torch.randn(*shape, device="cuda")
            yg = torch.randn(*shape, device="cuda")
            torch.cuda.synchronize()
            bench(lambda: xg + yg, f"cuda add {shape} (launch only)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--max-device-only":
        run_max_device_benchmarks()
        sys.exit(0)

    for flag in ("1", "0"):
        env = os.environ | {"TORCH_MAX_BACKEND_FAST_EAGER": flag}
        subprocess.run(
            [sys.executable, __file__, "--max-device-only"], env=env, check=True
        )
    run_torch_references()
