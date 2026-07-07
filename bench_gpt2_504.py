"""GPT-2 text generation latency: cuda vs max_device (fast eager path).

Real autoregressive run: greedy-decode N new tokens from a prompt with the
KV cache enabled, timed end-to-end (until the generated ids are back on
host), and print the generated text so you can see the model is doing the
real thing.

Usage: uv run python bench_gpt2_504.py [n_new_tokens]
"""

import statistics
import sys
import time

import torch

from torch_max_backend import register_max_devices

register_max_devices()

N_NEW_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 500
PROMPT = "In a shocking finding, scientists discovered a herd of unicorns"
WARMUP = 1
ITERS = 3


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids

    for dev in ["cuda", "max_device"]:
        m = model.to(dev)
        x = ids.to(dev)

        def step():
            with torch.no_grad():
                out = m.generate(
                    x,
                    max_new_tokens=N_NEW_TOKENS,
                    min_new_tokens=N_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            return out.to("cpu")

        for _ in range(WARMUP):
            out = step()
        times = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            out = step()
            times.append(time.perf_counter() - t0)
        mean_s = statistics.mean(times)
        n_generated = out.shape[1] - x.shape[1]
        print(f"\n===== {dev} =====", flush=True)
        print(
            f"generate {n_generated} tokens: mean={mean_s:.2f} s  "
            f"min={min(times):.2f} s  ({n_generated / mean_s:.1f} tok/s)",
            flush=True,
        )
        print(f"--- generated text ---\n{tok.decode(out[0])}", flush=True)
        model.to("cpu")


if __name__ == "__main__":
    main()
