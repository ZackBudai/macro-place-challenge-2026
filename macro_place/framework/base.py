"""Base classes for competition placers.

The framework keeps the submission contract small:
- implement ``initialize`` and optionally override later phases
- return a [num_macros, 2] tensor of center coordinates
- the framework restores fixed macros and can validate the result
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from macro_place.benchmark import Benchmark
from macro_place.framework.geometry import clamp_placement_to_canvas, seed_everything
from macro_place.utils import validate_placement


@dataclass
class PlacerConfig:
    """Shared configuration for framework-based placers."""

    seed: int = 42
    safety_gap: float = 1.0e-3
    optimize_soft_macros: bool = False
    validate_final_placement: bool = True


@dataclass
class PlacementResult:
    """Placement together with optional diagnostics."""

    placement: torch.Tensor
    valid: bool = True
    violations: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class CompetitionPlacer(ABC):
    """Template base class for competition submissions.

    Subclasses should override :meth:`initialize` and may optionally override
    :meth:`refine_hard_macros`, :meth:`refine_soft_macros`, or :meth:`finalize`.
    """

    def __init__(self, config: Optional[PlacerConfig] = None):
        self.config = config or PlacerConfig()
        self.last_result = None

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """Run the placement pipeline and return a legal placement tensor."""
        seed_everything(self.config.seed)

        placement = benchmark.macro_positions.clone()
        placement = self.initialize(benchmark, placement)
        placement = self.refine_hard_macros(benchmark, placement)

        if self.config.optimize_soft_macros:
            placement = self.refine_soft_macros(benchmark, placement)

        placement = self.finalize(benchmark, placement)
        placement = clamp_placement_to_canvas(
            placement,
            benchmark,
            gap=self.config.safety_gap,
            preserve_fixed=True,
        )

        valid, violations = validate_placement(placement, benchmark)
        result = PlacementResult(placement=placement, valid=valid, violations=violations)
        self.last_result = result

        if self.config.validate_final_placement and not valid:
            raise ValueError("Invalid placement: " + "; ".join(violations))

        return placement

    def initialize(self, benchmark: Benchmark, placement: torch.Tensor) -> torch.Tensor:
        """Create an initial placement. Subclasses must override this."""
        raise NotImplementedError

    def refine_hard_macros(
        self, benchmark: Benchmark, placement: torch.Tensor
    ) -> torch.Tensor:
        """Optional hard-macro refinement hook."""
        return placement

    def refine_soft_macros(
        self, benchmark: Benchmark, placement: torch.Tensor
    ) -> torch.Tensor:
        """Optional soft-macro refinement hook."""
        return placement

    def finalize(self, benchmark: Benchmark, placement: torch.Tensor) -> torch.Tensor:
        """Final cleanup hook before validation."""
        return placement
