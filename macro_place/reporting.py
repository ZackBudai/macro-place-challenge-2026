"""Reporting helpers for comparing the current solution to published stats."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.leaderboard import (
    IBM_BENCHMARKS,
    REPLACE_BASELINES,
    README_LEADERBOARD,
    SA_BASELINES,
    leaderboard_rank_for_score,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
TESTCASE_ROOT = REPO_ROOT / "external/MacroPlacement/Testcases/ICCAD04"


def evaluate_current_solution(
    placer=None,
    benchmark_names: Sequence[str] = IBM_BENCHMARKS,
    testcase_root: Path = TESTCASE_ROOT,
) -> List[Dict[str, object]]:
    """Run the current placer on the requested benchmarks and collect metrics."""

    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    from macro_place.utils import validate_placement
    from submissions.framework_example import FrameworkExamplePlacer

    if placer is None:
        placer = FrameworkExamplePlacer()

    results: List[Dict[str, object]] = []
    for name in benchmark_names:
        benchmark_dir = testcase_root / name
        if not benchmark_dir.exists():
            raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")

        benchmark, plc = load_benchmark_from_dir(str(benchmark_dir))
        import time

        begin = time.time()
        placement = placer.place(benchmark)
        runtime = time.time() - begin

        valid, violations = validate_placement(placement, benchmark)
        costs = compute_proxy_cost(placement, benchmark, plc)

        results.append(
            {
                "name": name,
                "proxy_cost": float(costs["proxy_cost"]),
                "wirelength": float(costs["wirelength_cost"]),
                "density": float(costs["density_cost"]),
                "congestion": float(costs["congestion_cost"]),
                "overlaps": int(costs["overlap_count"]),
                "runtime": float(runtime),
                "valid": bool(valid),
                "violations": violations,
                "sa_baseline": SA_BASELINES.get(name),
                "replace_baseline": REPLACE_BASELINES.get(name),
                "placement": placement,
                "benchmark": benchmark,
                "plc": plc,
            }
        )

    return results


def summarize_results(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate the current solution metrics into a report summary."""

    if not results:
        raise ValueError("No benchmark results provided")

    avg_proxy = sum(float(row["proxy_cost"]) for row in results) / len(results)
    avg_wl = sum(float(row["wirelength"]) for row in results) / len(results)
    avg_density = sum(float(row["density"]) for row in results) / len(results)
    avg_congestion = sum(float(row["congestion"]) for row in results) / len(results)
    avg_runtime = sum(float(row["runtime"]) for row in results) / len(results)
    total_overlaps = sum(int(row["overlaps"]) for row in results)
    valid_count = sum(1 for row in results if row["valid"])

    sa_values = [float(row["sa_baseline"]) for row in results if row.get("sa_baseline") is not None]
    replace_values = [
        float(row["replace_baseline"]) for row in results if row.get("replace_baseline") is not None
    ]
    avg_sa = sum(sa_values) / len(sa_values) if sa_values else None
    avg_replace = sum(replace_values) / len(replace_values) if replace_values else None

    leaderboard_rank = leaderboard_rank_for_score(avg_proxy)
    leaderboard_rows = [row for row in README_LEADERBOARD if isinstance(row.get("avg_proxy_cost"), (int, float))]
    best_readme = min(float(row["avg_proxy_cost"]) for row in leaderboard_rows)

    summary = {
        "num_benchmarks": len(results),
        "avg_proxy_cost": avg_proxy,
        "avg_wirelength": avg_wl,
        "avg_density": avg_density,
        "avg_congestion": avg_congestion,
        "avg_runtime_s": avg_runtime,
        "total_overlaps": total_overlaps,
        "valid_benchmarks": valid_count,
        "all_valid": valid_count == len(results),
        "avg_sa": avg_sa,
        "avg_replace": avg_replace,
        "leaderboard_rank": leaderboard_rank,
        "leaderboard_total": len(leaderboard_rows),
        "best_readme_avg_proxy_cost": best_readme,
        "gap_to_best_readme": avg_proxy - best_readme,
        "gap_to_replace": avg_proxy - float(avg_replace) if avg_replace is not None else None,
        "gap_to_sa": avg_proxy - float(avg_sa) if avg_sa is not None else None,
    }
    return summary


def write_report_markdown(summary: Dict[str, object], output_path: Path) -> None:
    """Write a concise Markdown summary of the current solution."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Current Solution Report",
        "",
        f"- Benchmarks evaluated: {summary['num_benchmarks']}",
        f"- Average proxy cost: {summary['avg_proxy_cost']:.4f}",
        f"- Average wirelength: {summary['avg_wirelength']:.4f}",
        f"- Average density: {summary['avg_density']:.4f}",
        f"- Average congestion: {summary['avg_congestion']:.4f}",
        f"- Average runtime: {summary['avg_runtime_s']:.2f}s/benchmark",
        f"- Overlaps: {summary['total_overlaps']}",
        f"- Valid benchmarks: {summary['valid_benchmarks']}/{summary['num_benchmarks']}",
        f"- README leaderboard rank: {summary['leaderboard_rank']} of {summary['leaderboard_total']}",
        f"- Gap to best README score: {summary['gap_to_best_readme']:+.4f}",
    ]

    if summary.get("avg_replace") is not None:
        lines.append(f"- Gap to RePlAce: {summary['gap_to_replace']:+.4f}")
    if summary.get("avg_sa") is not None:
        lines.append(f"- Gap to SA: {summary['gap_to_sa']:+.4f}")

    output_path.write_text("\n".join(lines) + "\n")


def _readme_comparison_rows(current_avg: float, top_n: int = 8) -> List[Dict[str, object]]:
    leaderboard_rows = [
        row
        for row in README_LEADERBOARD
        if isinstance(row.get("avg_proxy_cost"), (int, float))
    ]
    sorted_rows = sorted(leaderboard_rows, key=lambda row: float(row["avg_proxy_cost"]))
    selected = sorted_rows[:top_n]
    for baseline_name in ("RePlAce (baseline)", "SA (baseline)"):
        baseline_row = next(
            row for row in leaderboard_rows if str(row["team"]) == baseline_name
        )
        if baseline_row not in selected:
            selected.append(baseline_row)
    selected.append(
        {
            "team": "Current solution",
            "avg_proxy_cost": current_avg,
            "kind": "current",
        }
    )
    selected.sort(key=lambda row: float(row["avg_proxy_cost"]))
    return selected


def plot_leaderboard_comparison(
    summary: Dict[str, object],
    output_path: Path,
    top_n: int = 8,
) -> None:
    """Render a leaderboard-style comparison bar chart."""

    rows = _readme_comparison_rows(float(summary["avg_proxy_cost"]), top_n=top_n)
    labels = []
    values = []
    colors = []

    for row in rows:
        labels.append(str(row["team"]))
        values.append(float(row["avg_proxy_cost"]))
        if row.get("kind") == "current":
            colors.append("#f28e2b")
        elif "baseline" in str(row["team"]).lower():
            colors.append("#7f8c8d")
        else:
            colors.append("#4e79a7")

    fig_height = max(6.0, 0.45 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    y_pos = np.arange(len(rows))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Average proxy cost (lower is better)")
    ax.set_title("Current solution vs README leaderboard")
    ax.grid(axis="x", alpha=0.25)

    current_rank = int(summary["leaderboard_rank"])
    current_avg = float(summary["avg_proxy_cost"])
    ax.axvline(current_avg, color="#f28e2b", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(
        0.99,
        0.98,
        f"Current avg {current_avg:.4f} | rank {current_rank}",
        transform=ax.transAxes,
        color="#f28e2b",
        fontsize=10,
        ha="right",
        va="top",
        fontweight="bold",
    )

    for bar, value in zip(bars, values):
        ax.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(0, max(values + [current_avg]) * 1.08)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_comparison(results: Sequence[Dict[str, object]], output_path: Path) -> None:
    """Render per-benchmark proxy cost comparisons against the published baselines."""

    ordered = list(results)
    labels = [row["name"] for row in ordered]
    current = [float(row["proxy_cost"]) for row in ordered]
    sa = [float(row["sa_baseline"]) for row in ordered]
    replace = [float(row["replace_baseline"]) for row in ordered]

    x = np.arange(len(ordered))
    width = 0.26
    fig, ax = plt.subplots(figsize=(18, 6.5))
    ax.bar(x - width, current, width, label="Current solution", color="#f28e2b")
    ax.bar(x, sa, width, label="SA baseline", color="#bab0ab")
    ax.bar(x + width, replace, width, label="RePlAce baseline", color="#4e79a7")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Proxy cost")
    ax.set_title("Per-benchmark proxy cost vs README baselines")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    ymax = max(max(current), max(sa), max(replace))
    ax.set_ylim(0, ymax * 1.12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_improvement_summary(results: Sequence[Dict[str, object]], output_path: Path) -> None:
    """Render improvement percentages against SA and RePlAce for each benchmark."""

    labels = [row["name"] for row in results]
    vs_sa = [
        ((float(row["sa_baseline"]) - float(row["proxy_cost"])) / float(row["sa_baseline"])) * 100.0
        for row in results
    ]
    vs_replace = [
        (
            (float(row["replace_baseline"]) - float(row["proxy_cost"]))
            / float(row["replace_baseline"])
        )
        * 100.0
        for row in results
    ]

    x = np.arange(len(results))
    width = 0.38
    fig, ax = plt.subplots(figsize=(18, 5.5))
    ax.bar(x - width / 2, vs_sa, width, label="vs SA", color="#76b7b2")
    ax.bar(x + width / 2, vs_replace, width, label="vs RePlAce", color="#edc948")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Improvement (%)")
    ax.set_title("Current solution improvement over README baselines")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_placement_snapshots(
    results: Sequence[Dict[str, object]],
    output_dir: Path,
    benchmark_names: Iterable[str] | None = None,
    limit: int = 3,
) -> List[Path]:
    """Save a few detailed placement snapshots using the built-in visualizer."""

    from macro_place.utils import visualize_placement

    output_dir.mkdir(parents=True, exist_ok=True)
    selected: List[Dict[str, object]]

    if benchmark_names is not None:
        wanted = set(benchmark_names)
        selected = [row for row in results if row["name"] in wanted]
    else:
        ordered = sorted(results, key=lambda row: float(row["proxy_cost"]))
        if len(ordered) <= limit:
            selected = ordered
        else:
            candidates = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
            selected = []
            seen = set()
            for row in candidates:
                name = str(row["name"])
                if name not in seen:
                    selected.append(row)
                    seen.add(name)

    saved_paths: List[Path] = []
    for row in selected[:limit]:
        benchmark = row["benchmark"]
        placement = row["placement"]
        plc = row["plc"]
        safe_name = str(row["name"]).replace("/", "_")
        output_path = output_dir / f"{safe_name}.png"
        visualize_placement(placement, benchmark, save_path=str(output_path), plc=plc)
        saved_paths.append(output_path)

    return saved_paths


def build_report_bundle(
    results: Sequence[Dict[str, object]],
    output_dir: Path,
    snapshot_names: Iterable[str] | None = None,
    snapshot_limit: int = 3,
) -> Dict[str, object]:
    """Write the full report bundle and return the summary dictionary."""

    summary = summarize_results(results)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_report_markdown(summary, output_dir / "report.md")
    plot_leaderboard_comparison(summary, output_dir / "leaderboard_comparison.png")
    plot_benchmark_comparison(results, output_dir / "benchmark_comparison.png")
    plot_improvement_summary(results, output_dir / "improvement_summary.png")
    snapshot_paths = save_placement_snapshots(
        results,
        output_dir / "placements",
        benchmark_names=snapshot_names,
        limit=snapshot_limit,
    )

    summary["report_dir"] = str(output_dir)
    summary["snapshot_paths"] = [str(path) for path in snapshot_paths]
    return summary
