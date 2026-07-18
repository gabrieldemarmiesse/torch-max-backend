"""Compare matched functional groups from ROCm and Mojo GPT-2 profiles."""

import argparse
import csv
import json
from pathlib import Path

from tabulate import tabulate

GROUPS = [
    ("Transformer projections / MLP GEMMs", {"aten::addmm"}, {"aten::addmm"}),
    (
        "Attention over KV cache",
        {"aten::_flash_attention_forward"},
        {"aten::scaled_dot_product_attention"},
    ),
    ("KV-cache and sequence concat", {"aten::cat"}, {"aten::cat"}),
    ("LM-head GEMM", {"aten::mm"}, {"aten::linear"}),
    (
        "Elementwise GELU / residual",
        {"aten::add", "aten::mul", "aten::pow", "aten::tanh"},
        {"aten::add", "aten::mul", "aten::pow", "aten::tanh"},
    ),
    ("Tensor/logits copies", {"aten::copy_"}, {"aten::clone", "aten::_to_copy"}),
    ("Layer normalization", {"aten::native_layer_norm"}, {"aten::native_layer_norm"}),
    ("Logits processing", {"aten::where"}, {"aten::where"}),
    ("Greedy sampling", {"aten::argmax"}, {"aten::argmax"}),
]


def read_ops(path: Path) -> dict:
    with path.open() as file:
        return {
            row["aten_op"]: {
                "calls": int(row["calls"]),
                "us": float(row["self_gpu_time_us"]),
                "pct": float(row["self_gpu_pct"]),
            }
            for row in csv.DictReader(file)
        }


def group_values(ops: dict, names: set[str]) -> tuple[float, float]:
    return (
        sum(ops.get(name, {}).get("us", 0.0) for name in names),
        sum(ops.get(name, {}).get("pct", 0.0) for name in names),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rocm-dir", type=Path, default=Path("."))
    parser.add_argument("--mojo-dir", type=Path, default=Path("mojo_profile"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rocm_ops = read_ops(args.rocm_dir / "aten_gpu_time_decode.csv")
    mojo_ops = read_ops(args.mojo_dir / "aten_gpu_time_decode.csv")
    with (args.rocm_dir / "gpt2_aten_profile_summary.json").open() as file:
        rocm_summary = json.load(file)
    with (args.mojo_dir / "gpt2_aten_profile_summary.json").open() as file:
        mojo_summary = json.load(file)
    if rocm_summary["workload"] != mojo_summary["workload"]:
        raise RuntimeError("ROCm and Mojo workloads do not match")

    rows = []
    covered_rocm = set()
    covered_mojo = set()
    for group, rocm_names, mojo_names in GROUPS:
        rocm_us, rocm_pct = group_values(rocm_ops, rocm_names)
        mojo_us, mojo_pct = group_values(mojo_ops, mojo_names)
        covered_rocm.update(rocm_names)
        covered_mojo.update(mojo_names)
        rows.append(
            [
                group,
                "+".join(sorted(rocm_names)),
                rocm_us / 1000,
                rocm_pct,
                "+".join(sorted(mojo_names)),
                mojo_us / 1000,
                mojo_pct,
                mojo_us / rocm_us,
            ]
        )

    rocm_total = rocm_summary["profile"]["decode_self_gpu_us"]
    mojo_total = mojo_summary["profile"]["decode_self_gpu_us"]
    rocm_other = sum(
        value["us"] for name, value in rocm_ops.items() if name not in covered_rocm
    )
    mojo_other = sum(
        value["us"] for name, value in mojo_ops.items() if name not in covered_mojo
    )
    rows.append(
        [
            "Other generation/control ops",
            "remaining ATen ops",
            rocm_other / 1000,
            100 * rocm_other / rocm_total,
            "remaining ATen ops",
            mojo_other / 1000,
            100 * mojo_other / mojo_total,
            mojo_other / rocm_other,
        ]
    )

    print("\n================ decode: matched functional groups ================")
    print(
        tabulate(
            [
                [
                    row[0],
                    f"{row[2]:.3f}",
                    f"{row[3]:.2f}",
                    f"{row[5]:.3f}",
                    f"{row[6]:.2f}",
                    f"{row[7]:.2f}x",
                ]
                for row in rows
            ],
            headers=["function", "ROCm ms", "ROCm %", "Mojo ms", "Mojo %", "Mojo/ROCm"],
            tablefmt="simple",
            disable_numparse=True,
        )
    )

    with (args.output_dir / "gpt2_decode_function_comparison.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "function",
                "rocm_aten_ops",
                "rocm_self_gpu_ms",
                "rocm_decode_pct",
                "mojo_aten_ops",
                "mojo_self_gpu_ms",
                "mojo_decode_pct",
                "mojo_over_rocm_gpu_time",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row[0],
                    row[1],
                    f"{row[2]:.3f}",
                    f"{row[3]:.2f}",
                    row[4],
                    f"{row[5]:.3f}",
                    f"{row[6]:.2f}",
                    f"{row[7]:.3f}",
                ]
            )

    summary_rows = [
        (
            "unprofiled_wall_seconds",
            rocm_summary["unprofiled"]["wall_seconds"],
            mojo_summary["unprofiled"]["wall_seconds"],
        ),
        (
            "aggregate_tokens_per_second",
            rocm_summary["unprofiled"]["tokens_per_second"],
            mojo_summary["unprofiled"]["tokens_per_second"],
        ),
        (
            "prefill_self_gpu_ms",
            rocm_summary["profile"]["prefill_self_gpu_us"] / 1000,
            mojo_summary["profile"]["prefill_self_gpu_us"] / 1000,
        ),
        (
            "decode_self_gpu_ms",
            rocm_summary["profile"]["decode_self_gpu_us"] / 1000,
            mojo_summary["profile"]["decode_self_gpu_us"] / 1000,
        ),
        (
            "decode_self_gpu_ms_per_step",
            rocm_summary["profile"]["decode_ms_per_cached_step"],
            mojo_summary["profile"]["decode_ms_per_cached_step"],
        ),
        (
            "decode_trace_idle_pct",
            rocm_summary["profile"]["decode_gpu_trace"]["idle_pct"],
            mojo_summary["profile"]["decode_gpu_trace"]["idle_pct"],
        ),
    ]
    with (args.output_dir / "gpt2_backend_summary_comparison.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "rocm", "mojo", "mojo_over_rocm"])
        for metric, rocm_value, mojo_value in summary_rows:
            writer.writerow(
                [
                    metric,
                    f"{rocm_value:.6f}",
                    f"{mojo_value:.6f}",
                    f"{mojo_value / rocm_value:.6f}",
                ]
            )


if __name__ == "__main__":
    main()
