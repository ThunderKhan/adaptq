#!/usr/bin/env python3
"""Compare native kernel benchmark CSV results and flag regressions."""

from __future__ import annotations

import argparse
import csv
import sys
from typing import Dict, Tuple


Key = Tuple[str, str, str, str, str]


def load_results(path: str) -> Dict[Key, float]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        required = {
            "workload",
            "backend",
            "bits",
            "tokens",
            "padded",
            "ns_per_iteration",
        }
        if not required.issubset(set(rows.fieldnames or [])):
            raise ValueError("benchmark CSV is missing required columns")

        results = {}
        for row in rows:
            key = (
                row["workload"],
                row["backend"],
                row["bits"],
                row["tokens"],
                row["padded"],
            )
            results[key] = float(row["ns_per_iteration"])
        return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--max-regression", type=float, default=10.0)
    args = parser.parse_args()

    try:
        baseline = load_results(args.baseline)
        current = load_results(args.current)
    except (OSError, ValueError, KeyError) as exc:
        print(f"benchmark comparison failed: {exc}", file=sys.stderr)
        return 2

    regressions = 0
    compared = 0

    for key in sorted(current):
        if key not in baseline:
            continue

        before = baseline[key]
        after = current[key]
        if before <= 0.0 or after <= 0.0:
            print(f"{key}: invalid timing value", file=sys.stderr)
            return 2

        compared += 1
        change_pct = ((after / before) - 1.0) * 100.0
        print(
            f"{key}: {before:.3f} -> {after:.3f} ns/iter "
            f"({change_pct:+.2f}%)"
        )

        if change_pct > args.max_regression:
            regressions += 1

    if compared == 0:
        print("No matching benchmark rows were found.", file=sys.stderr)
        return 2

    if regressions:
        print(
            f"{regressions} benchmark regression(s) exceeded "
            f"{args.max_regression:.2f}%",
            file=sys.stderr,
        )
        return 1

    print(
        f"Benchmark regression check passed: {compared} matching rows, "
        f"threshold={args.max_regression:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
