#!/usr/bin/env python3
"""Generate visual comparisons for the current solution against the README leaderboard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running this script directly from scripts/ without installing the package.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from macro_place.leaderboard import IBM_BENCHMARKS
from macro_place.reporting import build_report_bundle, evaluate_current_solution
from submissions.framework_example import FrameworkExamplePlacer


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a report for the current solution")
    parser.add_argument(
        "--output-dir",
        default="reports/current_solution",
        help="Directory for the generated report artifacts.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=IBM_BENCHMARKS,
        help="Optional benchmark list to evaluate (defaults to the IBM suite).",
    )
    parser.add_argument(
        "--snapshot-names",
        nargs="*",
        default=None,
        help="Optional benchmark names to snapshot with the built-in placement visualizer.",
    )
    parser.add_argument(
        "--snapshot-limit",
        type=int,
        default=3,
        help="Maximum number of placement snapshots to save.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed passed to the framework example placer.",
    )
    args = parser.parse_args()

    placer = FrameworkExamplePlacer(seed=args.seed)
    try:
        results = evaluate_current_solution(placer=placer, benchmark_names=args.benchmarks)
    except ModuleNotFoundError as error:
        if "plc_client_os" not in str(error):
            raise
        print("Error: the TILOS MacroPlacement submodule is not initialized.")
        print("Run: git submodule update --init external/MacroPlacement")
        return 1

    summary = build_report_bundle(
        results,
        Path(args.output_dir),
        snapshot_names=args.snapshot_names,
        snapshot_limit=args.snapshot_limit,
    )

    serializable = {key: value for key, value in summary.items() if key != "snapshot_paths"}
    (Path(args.output_dir) / "summary.json").write_text(json.dumps(serializable, indent=2))

    print(f"Wrote report to {args.output_dir}")
    print(f"Average proxy cost: {summary['avg_proxy_cost']:.4f}")
    print(f"Leaderboard rank: {summary['leaderboard_rank']} of {summary['leaderboard_total']}")
    if summary.get("snapshot_paths"):
        print("Snapshots:")
        for path in summary["snapshot_paths"]:
            print(f"  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
