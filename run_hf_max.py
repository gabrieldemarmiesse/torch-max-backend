"""Smoke-test common HuggingFace models on max_device to find op coverage gaps.

Runs a forward pass on CPU and on max_device, compares outputs (loose tolerance
since we only care about op coverage, not bit-exactness), and reports a summary.

Usage:
    uv run python run_hf_max.py            # run all models
    uv run python run_hf_max.py bert gpt2  # run a subset
"""

import sys
import traceback

import torch

from torch_max_backend import register_max_devices

register_max_devices()

DEVICE = "max_device"
ATOL, RTOL = 2e-2, 2e-2


def _compare(model, inputs, extract):
    """Run on CPU then max_device and compare the extracted output tensor."""
    model = model.eval()
    with torch.no_grad():
        ref = extract(model(**inputs))

    model.to(DEVICE)
    max_inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out = extract(model(**max_inputs)).to("cpu")

    diff = (out - ref).abs()
    rel = diff / ref.abs().clamp_min(1e-6)
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = rel.max().item()
    # For logit outputs, also check the top-1 prediction agrees.
    argmax_match = torch.equal(ref.flatten(-1).argmax(-1), out.flatten(-1).argmax(-1))
    print(
        f"    max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}  "
        f"max_rel={max_rel:.3e}  argmax_match={argmax_match}",
        flush=True,
    )

    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)
    return (
        f"shape {tuple(out.shape)}, max_abs={max_abs:.2e}, argmax_match={argmax_match}"
    )


def run_bert():
    from transformers import AutoModel, AutoTokenizer

    name = "bert-base-uncased"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    inputs = tok("Hello world, this is a test.", return_tensors="pt")
    return name, _compare(model, inputs, lambda o: o.last_hidden_state)


def run_gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "gpt2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    inputs = tok("The quick brown fox", return_tensors="pt")
    return name, _compare(model, inputs, lambda o: o.logits)


def run_distilbert():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = "distilbert-base-uncased-finetuned-sst-2-english"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    inputs = tok("I really enjoyed this movie!", return_tensors="pt")
    return name, _compare(model, inputs, lambda o: o.logits)


def run_vit():
    from transformers import AutoModelForImageClassification

    name = "google/vit-base-patch16-224"
    model = AutoModelForImageClassification.from_pretrained(name)
    inputs = {"pixel_values": torch.randn(1, 3, 224, 224)}
    return name, _compare(model, inputs, lambda o: o.logits)


def run_resnet():
    from transformers import AutoModelForImageClassification

    name = "microsoft/resnet-18"
    model = AutoModelForImageClassification.from_pretrained(name)
    inputs = {"pixel_values": torch.randn(1, 3, 224, 224)}
    return name, _compare(model, inputs, lambda o: o.logits)


MODELS = {
    "bert": run_bert,
    "gpt2": run_gpt2,
    "distilbert": run_distilbert,
    "vit": run_vit,
    "resnet": run_resnet,
}


if __name__ == "__main__":
    keys = sys.argv[1:] or list(MODELS)
    results = {}
    for key in keys:
        print(f"\n===== {key} =====", flush=True)
        try:
            name, shape = MODELS[key]()
            results[key] = f"OK  ({name}, output {shape})"
            print(f"[OK] {results[key]}", flush=True)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            results[key] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"
            print(f"[FAIL] {key}: {results[key]}", flush=True)

    print("\n===== SUMMARY =====")
    for key in keys:
        print(f"  {key:12s} {results[key]}")
