"""Framework example placer upgraded to a stronger SA + legalization strategy."""

from __future__ import annotations

import torch

from macro_place.benchmark import Benchmark
from submissions.will_seed.placer import WillSeedPlacer


class FrameworkExamplePlacer:
    """Default report placer that reuses the stronger WillSeed strategy."""

    def __init__(self, seed: int = 42, refine_iters: int = 3200):
        self.seed = seed
        self.refine_iters = refine_iters
        self._delegate = WillSeedPlacer(seed=seed, refine_iters=refine_iters)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        return self._delegate.place(benchmark)
