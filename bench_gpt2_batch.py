"""GPT-2 batched generation throughput: cuda vs max_device (fast eager path).

Same greedy-decode loop as bench_gpt2_504.py but with a batch of identical
prompts, to see how much of the max_device gap is per-step host overhead
(python dispatch + device sync latency) that amortizes across the batch.

Usage: uv run python bench_gpt2_batch.py [n_new_tokens] [batch_sizes...]
       e.g. uv run python bench_gpt2_batch.py 200 1 8 32
"""

import statistics
import sys
import time

import torch

from torch_max_backend import register_max_devices

register_max_devices()

N_NEW_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
BATCHES = [int(b) for b in sys.argv[2:]] or [1, 8, 32]
PROMPT = "In a shocking finding, scientists discovered a herd of unicorns"
WARMUP = 1
ITERS = 3


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids

    results = {}
    for dev in ["cuda", "max_device"]:
        m = model.to(dev)
        for batch in BATCHES:
            x = ids.repeat(batch, 1).to(dev)

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
            n_generated = (out.shape[1] - x.shape[1]) * batch
            tok_s = n_generated / mean_s
            results[(dev, batch)] = (tok_s, out)
            print(
                f"{dev:>10}  batch={batch:<3} {n_generated} tokens: "
                f"mean={mean_s:.2f} s  ({tok_s:.1f} tok/s aggregate, "
                f"{tok_s / batch:.1f} tok/s/seq)",
                flush=True,
            )
        model.to("cpu")

    print()
    for batch in BATCHES:
        cuda_out = results[("cuda", batch)][1]
        max_out = results[("max_device", batch)][1]
        match = torch.equal(cuda_out, max_out)
        ratio = results[("max_device", batch)][0] / results[("cuda", batch)][0]
        print(
            f"batch={batch:<3} max_device/cuda = {ratio:.2f}x   "
            f"outputs identical: {match}"
        )
    print(f"\n--- sample (batch row 0) ---\n{tok.decode(results[('max_device', BATCHES[-1])][1][0])}")


if __name__ == "__main__":
    main()
