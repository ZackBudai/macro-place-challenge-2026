"""Tests for the reusable competition framework."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from macro_place import CompetitionPlacer, PlacerConfig, pack_macros_in_rows
from macro_place.utils import validate_placement


TESTCASE_ROOT = Path("external/MacroPlacement/Testcases/ICCAD04")


class _TestFrameworkPlacer(CompetitionPlacer):
    def __init__(self):
        super().__init__(PlacerConfig(validate_final_placement=True))

    def initialize(self, benchmark, placement):
        return pack_macros_in_rows(benchmark, placement=placement, gap=self.config.safety_gap)


@pytest.fixture
def ibm01():
    path = TESTCASE_ROOT / "ibm01"
    if not path.exists():
        pytest.skip("TILOS submodule not initialized")
    try:
        from macro_place import load_benchmark_from_dir

        return load_benchmark_from_dir(str(path))
    except ModuleNotFoundError:
        pytest.skip("TILOS submodule not initialized")


def test_pack_macros_in_rows_returns_tensor(ibm01):
    benchmark, _ = ibm01
    placement = pack_macros_in_rows(benchmark)
    assert placement.shape == (benchmark.num_macros, 2)


def test_framework_placer_produces_valid_result(ibm01):
    benchmark, _ = ibm01
    placer = _TestFrameworkPlacer()
    placement = placer.place(benchmark)
    is_valid, violations = validate_placement(placement, benchmark)
    assert is_valid, violations
