#!/usr/bin/env python3
"""Run the current solution on the processed public benchmarks and save results.

By default this executes the framework example placer on all benchmark tensors
in `benchmarks/processed/public` and writes a JSON summary. A CSV output mode is
also available for spreadsheet workflows.

Examples:
    python scripts/run_current_solution.py --output results.json
    python scripts/run_current_solution.py --output results.csv --format csv
    python scripts/run_current_solution.py --benchmarks ibm01 ibm02 --output subset.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_place.benchmark import Benchmark
from macro_place.utils import validate_placement
from submissions.framework_example import FrameworkExamplePlacer


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK_DIR = REPO_ROOT / "benchmarks/processed/public"


def _load_benchmark(path: Path) -> Benchmark:
    return Benchmark.load(str(path))


def _collect_benchmark_files(benchmark_names: List[str] | None) -> List[Path]:
    if benchmark_names:
        files = []
        for name in benchmark_names:
            path = DEFAULT_BENCHMARK_DIR / f"{name}.pt"
            if not path.exists():
                raise FileNotFoundError(f"Benchmark file not found: {path}")
            files.append(path)
        return files

    return sorted(DEFAULT_BENCHMARK_DIR.glob("*.pt"))


def _run_one(placer: FrameworkExamplePlacer, path: Path) -> Dict[str, object]:
    benchmark = _load_benchmark(path)

    start = time.time()
    placement = placer.place(benchmark)
    runtime_s = time.time() - start

    valid, violations = validate_placement(placement, benchmark)

    return {
        "file": path.name,
        "benchmark": benchmark.name,
        "num_macros": benchmark.num_macros,
        "status": "ok" if valid else "invalid",
        "valid": valid,
        "violations": violations,
        "runtime_s": runtime_s,
        "shape": list(placement.shape),
        "placement": placement,
    }


def _write_json(results: List[Dict[str, object]], output_path: Path) -> None:
    serializable = []
    for row in results:
        serializable.append(
            {
                key: value
                for key, value in row.items()
                if key != "placement"
            }
        )

    summary = {
        "total": len(serializable),
        "ok": sum(1 for row in serializable if row["status"] == "ok"),
        "invalid": sum(1 for row in serializable if row["status"] != "ok"),
        "results": serializable,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))


def _write_csv(results: List[Dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "file",
        "benchmark",
        "num_macros",
        "status",
        "valid",
        "runtime_s",
        "violations",
        "shape",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "file": row["file"],
                    "benchmark": row["benchmark"],
                    "num_macros": row["num_macros"],
                    "status": row["status"],
                    "valid": row["valid"],
                    "runtime_s": f"{row['runtime_s']:.6f}",
                    "violations": " | ".join(row["violations"]),
                    "shape": "x".join(str(dim) for dim in row["shape"]),
                }
            )


def _save_placements(results: List[Dict[str, object]], placement_dir: Path) -> None:
    placement_dir.mkdir(parents=True, exist_ok=True)
    for row in results:
        placement_path = placement_dir / f"{Path(row['file']).stem}_placement.pt"
        torch.save(row["placement"], placement_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the current macro placement solution")
    parser.add_argument(
        "--output",
        required=True,
        help="Output file path (.json or .csv).",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format. Default: json.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="Optional benchmark names to run (for example: ibm01 ibm02).",
    )
    parser.add_argument(
        "--save-placements",
        default=None,
        help="Optional directory to save each placement tensor as .pt files.",
    )
    args = parser.parse_args()

    benchmark_files = _collect_benchmark_files(args.benchmarks)
    placer = FrameworkExamplePlacer()

    results = []
    for path in benchmark_files:
        print(f"Running {path.name}...")
        results.append(_run_one(placer, path))

    output_path = Path(args.output)
    if args.format == "json":
        _write_json(results, output_path)
    else:
        _write_csv(results, output_path)

    if args.save_placements:
        _save_placements(results, Path(args.save_placements))

    ok = sum(1 for row in results if row["status"] == "ok")
    invalid = sum(1 for row in results if row["status"] != "ok")
    print(f"Wrote {output_path} ({ok} ok, {invalid} invalid)")
    if args.save_placements:
        print(f"Saved placement tensors to {args.save_placements}")

    return 0 if invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())