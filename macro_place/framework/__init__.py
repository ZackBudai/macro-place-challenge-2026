"""Reusable framework helpers for macro placement competition submissions."""

from macro_place.framework.base import CompetitionPlacer, PlacementResult, PlacerConfig
from macro_place.framework.geometry import (
    clamp_placement_to_canvas,
    macro_bbox,
    pack_macros_in_rows,
    seed_everything,
)
from macro_place.framework.suites import IBM_BENCHMARKS, NG45_BENCHMARKS

__all__ = [
    "CompetitionPlacer",
    "PlacementResult",
    "PlacerConfig",
    "clamp_placement_to_canvas",
    "macro_bbox",
    "pack_macros_in_rows",
    "seed_everything",
    "IBM_BENCHMARKS",
    "NG45_BENCHMARKS",
]
