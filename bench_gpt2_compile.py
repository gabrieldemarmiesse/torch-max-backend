"""GPT-2 batch-1 greedy decode: eager vs torch.compile, mojo vs cuda.

Each configuration should run in its own process (dynamo caches, the MAX
engine, and cuda contexts interact across configs).

Usage: uv run python bench_gpt2_compile.py <device> <mode> [n_new_tokens]
  device: mojo | cuda
  mode:
    eager            no compilation
    compile-max      model.forward compiled with max_backend
    compile-default  model.forward compiled with the default backend (inductor)
    compile-max-eager-attn
                     max_backend with attn_implementation="eager". On cuda,
                     HF's default sdpa route pads the mask with symbolic-dim
                     modulo arithmetic that MAX dims don't support; explicit
                     matmul attention sidesteps it (the mojo route decomposes
                     sdpa to the same math anyway).
"""

import statistics
import sys
import time

import torch

from torch_max_backend import max_backend, register_max_devices

register_max_devices()

DEVICE = sys.argv[1]
MODE = sys.argv[2]
N_NEW_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
PROMPT = "Here is how quantum computing works: "
WARMUP = 2  # first warmup pays compilation, second settles dynamic shapes
ITERS = 3


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    model_kwargs = {}
    if MODE == "compile-max-eager-attn":
        model_kwargs["attn_implementation"] = "eager"
    model = (
        AutoModelForCausalLM.from_pretrained("gpt2", **model_kwargs).eval().to(DEVICE)
    )
    x = tok(PROMPT, return_tensors="pt").input_ids.to(DEVICE)

    if MODE in ("compile-max", "compile-max-eager-attn"):
        model.forward = torch.compile(model.forward, backend=max_backend)
    elif MODE == "compile-default":
        model.forward = torch.compile(model.forward)
    elif MODE != "eager":
        raise SystemExit(f"unknown mode {MODE}")

    def step():
        with torch.no_grad():
            out = model.generate(
                x,
                max_new_tokens=N_NEW_TOKENS,
                min_new_tokens=N_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        return out.to("cpu")

    t0 = time.perf_counter()
    out = step()
    first_s = time.perf_counter() - t0
    for _ in range(WARMUP - 1):
        out = step()

    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = step()
        times.append(time.perf_counter() - t0)
    mean_s = statistics.mean(times)
    n_generated = out.shape[1] - x.shape[1]
    tok_s = n_generated / mean_s
    print(
        f"RESULT {DEVICE:>5} {MODE:<16} {n_generated} tokens: "
        f"mean={mean_s:.3f} s  ({tok_s:.1f} tok/s)  "
        f"first_run={first_s:.1f} s",
        flush=True,
    )
    text = tok.decode(out[0][x.shape[1] :])
    print(f"sample: {text[:120]!r}")


if __name__ == "__main__":
    main()
