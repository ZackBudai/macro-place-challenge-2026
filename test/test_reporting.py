"""Tests for the current-solution reporting helpers."""

import pytest

from macro_place.leaderboard import leaderboard_rank_for_score
from macro_place.reporting import summarize_results


def test_leaderboard_rank_for_score_inserts_before_best_entry():
    assert leaderboard_rank_for_score(1.3200) == 1


def test_summarize_results_aggregates_metrics():
    results = [
        {
            "proxy_cost": 1.0,
            "wirelength": 0.2,
            "density": 0.4,
            "congestion": 0.6,
            "overlaps": 0,
            "runtime": 1.5,
            "valid": True,
            "sa_baseline": 2.0,
            "replace_baseline": 1.5,
        },
        {
            "proxy_cost": 2.0,
            "wirelength": 0.3,
            "density": 0.5,
            "congestion": 0.7,
            "overlaps": 1,
            "runtime": 2.5,
            "valid": False,
            "sa_baseline": 4.0,
            "replace_baseline": 3.5,
        },
    ]

    summary = summarize_results(results)

    assert summary["num_benchmarks"] == 2
    assert summary["avg_proxy_cost"] == pytest.approx(1.5)
    assert summary["avg_wirelength"] == pytest.approx(0.25)
    assert summary["avg_density"] == pytest.approx(0.45)
    assert summary["avg_congestion"] == pytest.approx(0.65)
    assert summary["avg_runtime_s"] == pytest.approx(2.0)
    assert summary["total_overlaps"] == 1
    assert summary["valid_benchmarks"] == 1
    assert summary["all_valid"] is False
    assert summary["avg_sa"] == pytest.approx(3.0)
    assert summary["avg_replace"] == pytest.approx(2.5)
