#!/usr/bin/env python3
"""Load a MAX Graph from an MLIR file and try to materialize it in an inference session."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import max.driver
from max import engine
from max.graph import Graph


def _build_devices() -> list[max.driver.Device]:
    devices: list[max.driver.Device] = []

    # Prefer hardware accelerators if available.
    accelerator_count = getattr(max.driver, "accelerator_count", lambda: 0)()
    for i in range(accelerator_count):
        try:
            devices.append(max.driver.Accelerator(i))
        except Exception:
            # Keep probing all indices to preserve existing behavior in utility helpers.
            continue

    # Always provide a CPU fallback so the script can still run in CPU-only setups.
    devices.append(max.driver.CPU())
    return devices


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Max Graph from an MLIR file and attempt to load it into an "
            "engine inference session."
        )
    )
    parser.add_argument("mlir_path", type=Path, help="Path to the MLIR file to load")
    return parser.parse_args()


def _load_graph(path: Path) -> Graph:
    if not path.exists():
        raise FileNotFoundError(f"MLIR file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file path, got: {path}")

    graph = Graph(path=path, name="graph_from_mlir")
    return graph


def _load_graph_into_session(
    graph: Graph, devices: Iterable[max.driver.Device]
) -> None:
    session = engine.InferenceSession(devices=list(devices))
    session.load(graph)


def main():
    args = _parse_args()

    print(f"Loading graph from: {args.mlir_path}")
    graph = _load_graph(args.mlir_path)
    print(f"Created Graph: {graph}")

    devices = _build_devices()
    print(f"Creating InferenceSession with devices: {devices}")
    _load_graph_into_session(graph, devices)
    print("Session load succeeded.")


main()
