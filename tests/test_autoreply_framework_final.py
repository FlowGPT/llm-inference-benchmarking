from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.autoreply_framework_final as final


def champion(framework, qps, p50, *, fingerprint="same", passes=3):
    return {
        "framework": framework,
        "best_pass_qps": qps,
        "adjacent_fail_qps": round(qps + 0.1, 1),
        "invariant_fingerprint": fingerprint,
        "confirmations": [
            {"status": "PASS", "primary_p50": p50, "tail_p50": p50 + 0.05}
            for _ in range(passes)
        ],
    }


def test_final_order_is_balanced_and_each_framework_runs_three_times():
    assert final.interleaved_order() == [
        "vllm",
        "trtllm",
        "trtllm",
        "vllm",
        "vllm",
        "trtllm",
    ]


def test_rank_prefers_qps_then_latency_and_rejects_invariant_drift():
    vllm = champion("vllm", 10.1, 1.8)
    trtllm = champion("trtllm", 10.2, 1.9)

    assert final.select_winner([vllm, trtllm])["framework"] == "trtllm"
    trtllm["best_pass_qps"] = 10.1
    trtllm["adjacent_fail_qps"] = 10.2
    assert final.select_winner([vllm, trtllm])["framework"] == "vllm"

    trtllm["invariant_fingerprint"] = "different"
    with pytest.raises(ValueError, match="fingerprint"):
        final.select_winner([vllm, trtllm])


def test_qps_improvement_reports_absolute_and_relative_capacity_gain():
    gain = final.qps_improvement(baseline_qps=9.4, tuned_qps=10.1)

    assert gain == {
        "baseline_qps": 9.4,
        "tuned_qps": 10.1,
        "absolute_qps_gain": 0.7,
        "relative_percent_gain": 7.45,
    }
