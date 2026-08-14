from __future__ import annotations

import sys
from pathlib import Path

import json
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.autoreply_trtllm_tuning as tuning
import scripts.autoreply_vllm_tuning as vllm_tuning


def test_stable_image_and_pytorch_backend_preserve_exact_model_identity():
    candidate = tuning.Candidate(
        name="baseline",
        family="baseline",
        cli_args=(),
        llm_options={},
        hypothesis="resolved-default reference",
    )

    command = tuning.build_server_command(candidate)

    assert tuning.STABLE_IMAGE == "nvcr.io/nvidia/tensorrt-llm/release:1.2.1"
    assert tuning.RC_FALLBACK_IMAGE == "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22"
    assert command[command.index("--backend") + 1] == "pytorch"
    assert command[command.index("-e") + 1] == "PYTHONPATH=/opt/autoreply"
    assert command[command.index("--custom_module_dirs") + 1] == "/opt/autoreply"
    assert command[command.index("--custom_tokenizer") + 1] == (
        "trtllm_autoreply_tokenizer.AutoReplyTokenizer"
    )
    assert command[command.index("serve") + 1] == tuning.SERVED_MODEL
    assert command[command.index("--max_seq_len") + 1] == "8192"
    assert "trtllm-build" not in command
    assert "tensorrt" not in command[command.index("--backend") + 1]
    assert "--no-telemetry" not in command
    assert command[command.index("--name") + 1].startswith("autoreply-m12-trtllm-")


def test_rc_is_only_selected_after_stable_compatibility_failure():
    assert tuning.select_image(stable_compatible=True) == tuning.STABLE_IMAGE
    assert tuning.select_image(stable_compatible=False) == tuning.RC_FALLBACK_IMAGE


def test_candidate_config_covers_runtime_families_without_backend_override():
    candidates = tuning.load_candidates(
        tuning.WORKDIR / "configs/autoreply-trtllm-tuning-v121.json"
    )

    assert len(candidates) >= 20
    assert len({candidate.name for candidate in candidates}) == len(candidates)
    assert {candidate.family for candidate in candidates} >= {
        "batch_capacity",
        "token_capacity",
        "kv_cache",
        "scheduler",
        "cuda_graph",
        "frontend",
    }
    assert all("--backend" not in candidate.cli_args for candidate in candidates)
    assert all("--custom_tokenizer" not in candidate.cli_args for candidate in candidates)
    assert all("--num_serve_frontends" not in candidate.cli_args for candidate in candidates)


def test_candidate_cannot_override_fixed_max_sequence_length(tmp_path):
    path = tmp_path / "candidates.json"
    path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "name": "invalid",
                        "family": "invalid",
                        "cli_args": ["--max_seq_len", "4096"],
                        "llm_options": {},
                        "hypothesis": "must be rejected",
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="max_seq_len"):
        tuning.load_candidates(path)


def test_help_inventory_extracts_and_explicitly_classifies_flags():
    flags = tuning.extract_flags(
        "usage: trtllm-serve [--backend NAME] [--max_seq_len N] "
        "[--max_batch_size N] [--future_option X]"
    )
    rows = [tuning.classify_flag(flag) for flag in flags]

    assert flags == [
        "--backend",
        "--future_option",
        "--max_batch_size",
        "--max_seq_len",
    ]
    by_flag = {row["flag"]: row for row in rows}
    assert by_flag["--backend"]["class"] == "fixed-invariant"
    assert by_flag["--max_seq_len"]["class"] == "fixed-invariant"
    assert by_flag["--max_batch_size"]["class"] == "applicable-search"
    assert by_flag["--future_option"]["class"] == "review-required"
    assert all(row["reason"] for row in rows)


def test_trtllm_reuses_the_exact_fixed_replay_builder(tmp_path):
    assert tuning.build_replay_command is vllm_tuning.build_replay_command

    command = tuning.build_replay_command(
        qps=9.4,
        rounds=12,
        output=tmp_path / "probe.jsonl",
        api_key="local-key",
    )
    assert command[command.index("--max-tokens") + 1] == "50"
    assert command[command.index("--temperature") + 1] == "0.7"
    assert command[command.index("--top-p") + 1] == "0.8"
    assert json.loads(command[command.index("--extra-body-json") + 1]) == {
        "n": 3,
        "stop": ["<|im_end|>"],
        "top_k": -1,
    }


def test_trtllm_container_ownership_is_exact():
    assert tuning.is_owned_container("autoreply-m12-trtllm-baseline") is True
    assert tuning.is_owned_container("autoreply-m12-trtllm-") is False
    assert tuning.is_owned_container("autoreply-m12-vllm-baseline") is False
    assert tuning.is_owned_container("minio_test") is False
