"""Benchmark per-op-call overhead of mojo eager mode.

Usage:
    uv run python scripts/benchmark_fast_eager.py

Times a few elementwise ops on the mojo device, plus torch-native CPU and
CUDA references.
"""

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


def run_mojo_device_benchmarks():
    import torch

    from torch_mojo_backend import register_mojo_devices

    register_mojo_devices()
    print("  mojo eager mode")
    dev = torch.device("mojo")
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
    run_mojo_device_benchmarks()
    run_torch_references()
