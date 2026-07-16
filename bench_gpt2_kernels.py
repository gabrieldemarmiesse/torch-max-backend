"""Microbenchmark the batch-decode kernels used by GPT-2.

The benchmark keeps all inputs and outputs on the selected device and
synchronizes once around a group of launches.  This measures steady-state
kernel, allocation, and dispatch cost without adding a device-to-host copy to
every sample.

Usage: uv run python bench_gpt2_kernels.py <device> [batch] [kv_len]
       e.g. uv run python bench_gpt2_kernels.py mojo 256 128
            uv run python bench_gpt2_kernels.py cuda 256 128
"""

import statistics
import sys
import time

import torch
import torch.nn.functional as F

from torch_mojo_backend import get_accelerators, register_mojo_devices

register_mojo_devices()

DEVICE = sys.argv[1]
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 256
KV_LEN = int(sys.argv[3]) if len(sys.argv) > 3 else 128
WARMUP = 5
ITERS = 30
ROUNDS = 3


def _synchronize():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    else:
        list(get_accelerators())[int(DEVICE.partition(":")[2] or 0)].synchronize()


def _bench(fn):
    out = None
    for _ in range(WARMUP):
        out = fn()
    _synchronize()
    samples = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            out = fn()
        _synchronize()
        samples.append((time.perf_counter() - t0) * 1e3 / ITERS)
    return statistics.median(samples), out


def _randn(*shape):
    return torch.randn(*shape).to(DEVICE)


def main():
    cases = []

    def add_addmm(name, k, n):
        x = _randn(BATCH, k)
        weight = _randn(k, n)
        bias = _randn(n)
        cases.append((name, lambda: torch.addmm(bias, x, weight)))

    add_addmm("qkv addmm 768x2304", 768, 2304)
    add_addmm("attn proj addmm 768x768", 768, 768)
    add_addmm("mlp fc addmm 768x3072", 768, 3072)
    add_addmm("mlp proj addmm 3072x768", 3072, 768)

    hidden = _randn(BATCH, 768)
    lm_weight = _randn(50_257, 768)
    cases.append(("lm head linear 768x50257", lambda: F.linear(hidden, lm_weight)))

    q = _randn(BATCH, 12, 1, 64)
    key = _randn(BATCH, 12, KV_LEN, 64)
    value = _randn(BATCH, 12, KV_LEN, 64)
    cases.append(
        (
            f"decode attention kv={KV_LEN}",
            lambda: F.scaled_dot_product_attention(q, key, value),
        )
    )

    norm_input = _randn(BATCH, 768)
    norm_weight = _randn(768)
    norm_bias = _randn(768)
    cases.append(
        (
            "layer norm 768",
            lambda: F.layer_norm(norm_input, (768,), norm_weight, norm_bias),
        )
    )

    gelu_input = _randn(BATCH, 3072)
    cases.append(("gelu tanh 3072", lambda: F.gelu(gelu_input, approximate="tanh")))

    residual_a = _randn(BATCH, 768)
    residual_b = _randn(BATCH, 768)
    cases.append(("residual add 768", lambda: residual_a + residual_b))

    logits = _randn(BATCH, 50_257)
    cases.append(("argmax vocab 50257", lambda: torch.argmax(logits, dim=-1)))

    past_key = _randn(BATCH, 12, KV_LEN, 64)
    fused_qkv = _randn(BATCH, 1, 2304)
    new_key = fused_qkv[:, :, 768:1536].view(BATCH, 1, 12, 64).transpose(1, 2)
    cases.append(
        (f"kv cat strided kv={KV_LEN}", lambda: torch.cat((past_key, new_key), dim=-2))
    )

    print(f"device={DEVICE} batch={BATCH} kv_len={KV_LEN}", flush=True)
    for name, fn in cases:
        ms, out = _bench(fn)
        print(f"{name:<31} {ms:9.3f} ms", flush=True)
    del out


if __name__ == "__main__":
    main()
