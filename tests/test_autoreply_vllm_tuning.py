from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.autoreply_vllm_tuning as tuning


def test_replay_command_preserves_autoreply_contract(tmp_path):
    command = tuning.build_replay_command(
        qps=9.4,
        rounds=6,
        output=tmp_path / "probe.jsonl",
        api_key="local-test-key",
    )

    joined = " ".join(command)
    assert "--max-tokens 50" in joined
    assert "--temperature 0.7" in joined
    assert "--top-p 0.8" in joined
    assert "--frequency-penalty 0.01" in joined
    assert "--presence-penalty 0.01" in joined
    assert "--disable-min-p" in command
    assert json.loads(command[command.index("--extra-body-json") + 1]) == {
        "n": 3,
        "stop": ["<|im_end|>"],
        "top_k": -1,
    }
    assert "--preselected-route" in command
    assert "--sample-range" not in command


def test_server_command_preserves_model_and_uses_owned_container():
    candidate = tuning.Candidate(
        name="baseline",
        family="baseline",
        args=(),
        hypothesis="production reference",
    )

    command = tuning.build_server_command(candidate, api_key="local-test-key")

    assert tuning.container_name(candidate) == "autoreply-m12-vllm-baseline"
    assert command[command.index("--name") + 1] == tuning.container_name(candidate)
    assert command[command.index("--model") + 1] == tuning.MODEL_PATH
    assert command[command.index("--served-model-name") + 1] == tuning.SERVED_MODEL
    assert command[command.index("--quantization") + 1] == "modelopt"
    assert command[command.index("--kv-cache-dtype") + 1] == "fp8"
    assert command[command.index("--max-model-len") + 1] == "8192"
    assert command[command.index("--api-key") + 1] == "local-test-key"


def test_flag_inventory_has_explicit_disposition():
    flags = tuning.extract_flags(
        "usage: api_server.py [--max-num-seqs N] [--enable-lora] [--api-key K]"
    )

    rows = [tuning.classify_flag(flag) for flag in flags]

    assert {row["flag"] for row in rows} == {
        "--api-key",
        "--enable-lora",
        "--max-num-seqs",
    }
    assert all(row["class"] and row["reason"] for row in rows)
    assert tuning.classify_flag("--future-unknown")["class"] == "review-required"


def test_probe_shell_keeps_sampling_literal_and_accepts_named_run_settings():
    text = (tuning.WORKDIR / "scripts/run_autoreply_probe.sh").read_text()

    for literal in (
        "--max-tokens 50",
        "--temperature 0.7",
        "--top-p 0.8",
        "--frequency-penalty 0.01",
        "--presence-penalty 0.01",
        "--disable-min-p",
        "'{\"n\":3,\"stop\":[\"<|im_end|>\"],\"top_k\":-1}'",
    ):
        assert literal in text
    assert "AUTOREPLY_ROUNDS" in text
    assert "AUTOREPLY_TAIL_WINDOW" in text
    assert "AUTOREPLY_ARTIFACT_DIR" in text
    assert "sed -n 's/^API_KEY=" not in text


def test_probe_output_contract_rejects_empty_success_rounds(tmp_path):
    output = tmp_path / "probe.jsonl"
    output.write_text(
        json.dumps(
            {
                "measurement_round": 1,
                "success_rate": 100.0,
                "output_tokens": 0,
            }
        )
        + "\n"
    )

    result = tuning.validate_probe_output(output)

    assert result["status"] == "INVALID_OUTPUT"
    assert result["empty_output_rounds"] == [1]


def test_probe_output_contract_accepts_nonempty_rounds(tmp_path):
    output = tmp_path / "probe.jsonl"
    output.write_text(
        "\n".join(
            json.dumps({"measurement_round": round_id, "output_tokens": 100})
            for round_id in (1, 2)
        )
        + "\n"
    )

    result = tuning.validate_probe_output(output)

    assert result == {
        "status": "OK",
        "rounds_seen": 2,
        "total_output_tokens": 200,
        "empty_output_rounds": [],
    }


def test_binary_search_probes_low_before_high_and_finds_tenth_boundary():
    observed = []

    def probe(qps):
        observed.append(qps)
        return {"status": "PASS" if qps <= 9.7 else "FAIL", "qps": qps}

    result = tuning.binary_search(probe, low=8.5, high=10.5, precision=0.1)

    assert observed[:2] == [8.5, 10.5]
    assert result["best_pass_qps"] == 9.7
    assert result["adjacent_fail_qps"] == 9.8
    assert result["final_bracket"] == [9.7, 9.8]


def test_candidate_config_is_unique_and_cannot_override_invariants():
    candidates = tuning.load_candidates(
        tuning.WORKDIR / "configs/autoreply-vllm-tuning-v0271.json"
    )

    assert len(candidates) >= 20
    assert len({candidate.name for candidate in candidates}) == len(candidates)
    assert {candidate.family for candidate in candidates} >= {
        "sequence_capacity",
        "batch_tokens",
        "prefill",
        "gpu_memory",
        "kv_layout",
        "kernel",
        "cuda_graph_compile",
        "frontend",
    }
    forbidden = {"--model", "--served-model-name", "--quantization", "--api-key"}
    assert all(not forbidden.intersection(candidate.args) for candidate in candidates)


def test_redaction_and_container_ownership_are_exact():
    command = ["serve", "--api-key", "secret-value", "--model", "safe"]

    assert tuning.redact_command(command) == [
        "serve",
        "--api-key",
        "<redacted>",
        "--model",
        "safe",
    ]
    assert tuning.is_owned_container("autoreply-m12-vllm-baseline") is True
    assert tuning.is_owned_container("autoreply-m12-vllm-") is False
    assert tuning.is_owned_container("autoreply-m12-prod") is False


def test_combine_candidates_keeps_fixed_model_length_out_of_search_args():
    scheduled = tuning.Candidate(
        name="scheduled4096",
        family="scheduler",
        args=("--scheduler-delay-factor", "0.01"),
        hypothesis="scheduler candidate",
    )
    api = tuning.Candidate(
        name="api2",
        family="frontend",
        args=("--api-server-count", "2"),
        hypothesis="frontend candidate",
    )

    combined = tuning.combine_candidates(
        "scheduled4096-api2", scheduled, api
    )

    assert combined.args == scheduled.args + api.args
    assert "--max-model-len" not in combined.args
    assert combined.family == "interaction"


def test_search_candidate_cold_starts_each_qps_in_an_independent_directory(
    monkeypatch, tmp_path
):
    candidate = tuning.Candidate(
        name="winner",
        family="interaction",
        args=(),
        hypothesis="winner",
    )
    calls = []

    def fake_screen(candidate, *, qps, rounds, tail_window, run_root, keep=False):
        calls.append((qps, rounds, tail_window, run_root, keep))
        return {"candidate": candidate.name, "qps": qps, "status": "PASS" if qps <= 9.7 else "FAIL"}

    monkeypatch.setattr(tuning, "screen_candidate", fake_screen)

    result = tuning.search_candidate(
        candidate,
        low=9.6,
        high=9.8,
        run_root=tmp_path,
    )

    assert result["best_pass_qps"] == 9.7
    assert result["adjacent_fail_qps"] == 9.8
    assert [(qps, rounds, tail) for qps, rounds, tail, _, _ in calls] == [
        (9.6, 12, 6),
        (9.8, 12, 6),
        (9.7, 12, 6),
    ]
    assert len({str(run_root) for _, _, _, run_root, _ in calls}) == 3
    assert all(keep is False for _, _, _, _, keep in calls)
