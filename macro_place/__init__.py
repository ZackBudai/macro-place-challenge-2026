"""macro_place - Macro Placement Challenge toolkit.

The package keeps imports lightweight so framework utilities can be used even
when the optional TILOS MacroPlacement submodule is not present.
"""

from macro_place.benchmark import Benchmark
from macro_place.framework import (
    CompetitionPlacer,
    PlacementResult,
    PlacerConfig,
    clamp_placement_to_canvas,
    macro_bbox,
    pack_macros_in_rows,
    seed_everything,
)

__all__ = [
    "Benchmark",
    "CompetitionPlacer",
    "PlacementResult",
    "PlacerConfig",
    "clamp_placement_to_canvas",
    "macro_bbox",
    "pack_macros_in_rows",
    "seed_everything",
    "load_benchmark",
    "load_benchmark_from_dir",
    "compute_proxy_cost",
    "compute_overlap_metrics",
    "validate_placement",
    "visualize_placement",
]


def __getattr__(name):
    """Lazily import heavier helpers on demand."""
    if name in {"load_benchmark", "load_benchmark_from_dir"}:
        from macro_place.loader import load_benchmark, load_benchmark_from_dir

        return {"load_benchmark": load_benchmark, "load_benchmark_from_dir": load_benchmark_from_dir}[name]
    if name in {"compute_proxy_cost", "compute_overlap_metrics"}:
        from macro_place.objective import compute_proxy_cost, compute_overlap_metrics

        return {
            "compute_proxy_cost": compute_proxy_cost,
            "compute_overlap_metrics": compute_overlap_metrics,
        }[name]
    if name in {"validate_placement", "visualize_placement"}:
        from macro_place.utils import validate_placement, visualize_placement

        return {
            "validate_placement": validate_placement,
            "visualize_placement": visualize_placement,
        }[name]
    raise AttributeError(name)
