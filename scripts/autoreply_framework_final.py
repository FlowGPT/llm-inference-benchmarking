#!/usr/bin/env python3
"""Validation and ranking for the AutoReply cross-framework final."""

from __future__ import annotations

import statistics
from typing import Any


def qps_improvement(*, baseline_qps: float, tuned_qps: float) -> dict[str, float]:
    """Return capacity gain at the shared latency SLO."""
    if baseline_qps <= 0:
        raise ValueError("baseline_qps must be positive")
    absolute = round(tuned_qps - baseline_qps, 1)
    relative = round(absolute / baseline_qps * 100, 2)
    return {
        "baseline_qps": baseline_qps,
        "tuned_qps": tuned_qps,
        "absolute_qps_gain": absolute,
        "relative_percent_gain": relative,
    }


def interleaved_order() -> list[str]:
    """Balance first/second execution position across three cold runs."""
    return ["vllm", "trtllm", "trtllm", "vllm", "vllm", "trtllm"]


def _validate(champions: list[dict[str, Any]]) -> None:
    if not champions:
        raise ValueError("no champions supplied")
    fingerprints = {champion.get("invariant_fingerprint") for champion in champions}
    if None in fingerprints or len(fingerprints) != 1:
        raise ValueError("champion invariant fingerprint mismatch")
    for champion in champions:
        qps = champion.get("best_pass_qps")
        adjacent = champion.get("adjacent_fail_qps")
        if not isinstance(qps, (int, float)) or not isinstance(adjacent, (int, float)):
            raise ValueError("champion has no numeric QPS boundary")
        if round(adjacent - qps, 1) != 0.1:
            raise ValueError("champion has no adjacent 0.1-QPS failure")
        confirmations = champion.get("confirmations", [])
        if len(confirmations) != 3 or any(
            run.get("status") != "PASS" for run in confirmations
        ):
            raise ValueError("champion requires exactly three passing confirmations")


def select_winner(champions: list[dict[str, Any]]) -> dict[str, Any]:
    _validate(champions)

    def rank(champion: dict[str, Any]) -> tuple:
        confirmations = champion["confirmations"]
        primary = statistics.median(run["primary_p50"] for run in confirmations)
        tail = statistics.median(run["tail_p50"] for run in confirmations)
        error_rate = statistics.mean(run.get("error_rate", 0.0) for run in confirmations)
        variance = statistics.pvariance(
            run["primary_p50"] for run in confirmations
        )
        complexity = champion.get("non_default_parameter_count", 0)
        return (
            -champion["best_pass_qps"],
            primary,
            tail,
            error_rate,
            variance,
            complexity,
        )

    return min(champions, key=rank)
