"""GPT-2 batched generation throughput on one device (fast eager path).

Greedy-decodes 200 tokens from a fixed prompt repeated across the batch,
and reports aggregate and per-sequence throughput.

Usage: uv run python bench_gpt2_batch.py <device> [batch_sizes...]
       e.g. uv run python bench_gpt2_batch.py max_device 1 8 32
            uv run python bench_gpt2_batch.py cuda 256
"""

import statistics
import sys
import time

import torch

from torch_max_backend import register_max_devices

register_max_devices()

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "max_device"
BATCHES = [int(b) for b in sys.argv[2:]] or [256]
N_NEW_TOKENS = 200
PROMPT = "Here is how quantum computing works: "
WARMUP = 1
ITERS = 3


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval().to(DEVICE)
    ids = tok(PROMPT, return_tensors="pt").input_ids

    last_out = None
    for batch in BATCHES:
        x = ids.repeat(batch, 1).to(DEVICE)

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

        for _ in range(WARMUP):
            out = step()
        times = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            out = step()
            times.append(time.perf_counter() - t0)
        mean_s = statistics.mean(times)
        n_generated = (out.shape[1] - x.shape[1]) * batch
        tok_s = n_generated / mean_s
        last_out = out
        print(
            f"{DEVICE:>10}  batch={batch:<3} {n_generated} tokens: "
            f"mean={mean_s:.2f} s  ({tok_s:.1f} tok/s aggregate, "
            f"{tok_s / batch:.1f} tok/s/seq)",
            flush=True,
        )

    text = tok.decode(last_out[0]).replace(tok.eos_token, "<EOS>")
    print(f"\n--- sample (batch row 0) ---\n{text}")


if __name__ == "__main__":
    main()
