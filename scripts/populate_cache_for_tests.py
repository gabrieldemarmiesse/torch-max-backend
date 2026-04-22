#!/usr/bin/env python3
"""Run pytest tests in randomized batches.

Steps:
1) Collect all test nodeids.
2) Shuffle them.
3) Execute tests in batches (default 20 tests per batch).
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from dataclasses import dataclass

import pytest


def _collect_tests(collect_args: list[str]) -> list[str]:
    """Collect all test node ids via pytest's collect-only mode."""

    @dataclass
    class _Collector:
        nodeids: list[str]

    collector = _Collector(nodeids=[])

    class _Hook:
        def pytest_collection_finish(
            self, session
        ):  # pragma: no cover - plugin callback
            collector.nodeids = [item.nodeid for item in session.items]

    # Keep output minimal while collecting.
    exit_code = pytest.main(
        ["--collect-only", "-n", "3", "--disable-warnings", *collect_args],
        plugins=[_Hook()],
    )
    if exit_code != 0:
        raise RuntimeError(f"pytest collection failed with exit code {exit_code}")

    return collector.nodeids


def _run_batch(batch: list[str], run_args: list[str]) -> int:
    """Run one batch of pytest node ids and return the exit code."""

    cmd = [sys.executable, "-m", "pytest", "-n", "3", *run_args, *batch]
    proc = subprocess.run(cmd)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect all pytest tests, shuffle, and run in batches"
    )
    parser.add_argument(
        "paths", nargs="*", default=["."], help="Paths to collect tests from"
    )
    parser.add_argument(
        "--batch-size", type=int, default=20, help="Number of tests per batch"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for deterministic shuffling"
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Collect and print tests only, do not execute",
    )
    parser.add_argument(
        "--collect-only", action="store_true", help="Alias for --no-run"
    )
    args, extra_pytest_args = parser.parse_known_args()

    test_args = ["-q", *args.paths, *extra_pytest_args]
    test_node_ids = _collect_tests(test_args)

    if not test_node_ids:
        print("No tests collected.")
        return 0

    print("Collected tests:")
    for nodeid in test_node_ids:
        print(nodeid)

    if args.seed is not None:
        random.seed(args.seed)
    random.shuffle(test_node_ids)

    if args.no_run or args.collect_only:
        return 0

    if args.batch_size <= 0:
        raise ValueError("batch-size must be greater than 0")

    run_args = ["-q", "--disable-warnings", *extra_pytest_args]
    failed_batches = []

    for batch_index, start in enumerate(
        range(0, len(test_node_ids), args.batch_size), start=1
    ):
        batch = test_node_ids[start : start + args.batch_size]
        print(f"Running batch {batch_index}: {len(batch)} tests")
        code = _run_batch(batch, run_args)
        if code != 0:
            failed_batches.append(batch_index)

    if failed_batches:
        print(f"Batches failed: {', '.join(map(str, failed_batches))}")
        return 1

    print("All batches passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
