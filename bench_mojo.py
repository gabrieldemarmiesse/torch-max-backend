"""Benchmark forward-pass latency across cuda / mojo:0 (GPU) / mojo:1 (CPU).

Each forward is timed end-to-end including bringing the output back to host
(this also forces lazy mojo realization, and synchronizes CUDA).

Usage: uv run python bench_mojo.py [resnet bert ...]
"""

import statistics
import sys
import time

import torch

from torch_mojo_backend import register_mojo_devices

register_mojo_devices()

DEVICES = ["cuda", "mojo:0", "mojo:1"]
WARMUP = 3
ITERS = 20


def _bench_one(model, inputs, device):
    model = model.eval().to(device)
    dev_inputs = {k: v.to(device) for k, v in inputs.items()}

    def step():
        with torch.no_grad():
            out = model(**dev_inputs)
        out = out.logits if hasattr(out, "logits") else out.last_hidden_state
        out.to("cpu")  # force realization (mojo) / sync + D2H (cuda)

    for _ in range(WARMUP):
        step()
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        step()
        times.append(time.perf_counter() - t0)
    # free GPU memory before the next device
    model.to("cpu")
    return statistics.mean(times), min(times)


def _load_resnet():
    from transformers import AutoModelForImageClassification

    m = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
    return m, {"pixel_values": torch.randn(1, 3, 224, 224)}


def _load_bert():
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    m = AutoModel.from_pretrained("bert-base-uncased")
    return m, dict(
        tok("Hello world, this is a benchmark sentence.", return_tensors="pt")
    )


def _load_gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    m = AutoModelForCausalLM.from_pretrained("gpt2")
    return m, dict(tok("The quick brown fox jumps over", return_tensors="pt"))


LOADERS = {"resnet": _load_resnet, "bert": _load_bert, "gpt2": _load_gpt2}
LABEL = {
    "cuda": "cuda (torch GPU)",
    "mojo:0": "mojo:0 (MAX GPU)",
    "mojo:1": "mojo:1 (MAX CPU)",
}


if __name__ == "__main__":
    keys = sys.argv[1:] or list(LOADERS)
    results = {}
    for key in keys:
        print(f"\n===== {key} =====", flush=True)
        model, inputs = LOADERS[key]()
        results[key] = {}
        for dev in DEVICES:
            try:
                mean_s, min_s = _bench_one(model, inputs, dev)
                results[key][dev] = mean_s
                print(
                    f"  {LABEL[dev]:24s} mean={mean_s * 1e3:8.2f} ms  "
                    f"min={min_s * 1e3:8.2f} ms",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                results[key][dev] = None
                print(f"  {LABEL[dev]:24s} FAILED: {type(e).__name__}: {e}", flush=True)

    print("\n===== SUMMARY (mean ms, and speedup vs cuda) =====")
    for key in keys:
        base = results[key].get("cuda")
        row = []
        for dev in DEVICES:
            v = results[key][dev]
            if v is None:
                row.append(f"{dev}=FAIL")
            elif base:
                row.append(f"{dev}={v * 1e3:.2f}ms ({base / v:.2f}x)")
            else:
                row.append(f"{dev}={v * 1e3:.2f}ms")
        print(f"  {key:8s} " + "  ".join(row))
