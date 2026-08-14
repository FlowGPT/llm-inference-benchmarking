#!/usr/bin/env python3
"""TensorRT-LLM tuning adapter for the fixed AutoReply workload."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from scripts.autoreply_vllm_tuning import (
        MODEL_PATH,
        SERVED_MODEL,
        WORKDIR,
        build_replay_command,
    )
except ModuleNotFoundError:  # Direct execution from scripts/.
    from autoreply_vllm_tuning import (
        MODEL_PATH,
        SERVED_MODEL,
        WORKDIR,
        build_replay_command,
    )


STABLE_IMAGE = "nvcr.io/nvidia/tensorrt-llm/release:1.2.1"
RC_FALLBACK_IMAGE = "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22"
FIXED_CLI_FLAGS = {
    "--backend",
    "--custom_module_dirs",
    "--custom_tokenizer",
    "--max_seq_len",
}
SEARCH_CLI_FLAGS = {
    "--max_batch_size",
    "--max_num_tokens",
    "--kv_cache_free_gpu_memory_fraction",
    "--enable_chunked_prefill",
    "--num_postprocess_workers",
}


def extract_flags(help_text: str) -> list[str]:
    """Extract deterministic long-option inventory from the live image help."""
    return sorted(set(re.findall(r"(?<![\w-])--[a-zA-Z0-9][\w-]*", help_text)))


def classify_flag(flag: str) -> dict[str, str]:
    if flag in FIXED_CLI_FLAGS:
        return {
            "flag": flag,
            "class": "fixed-invariant",
            "reason": "fixed by the PyTorch-backend and 8,192-token experiment contract",
        }
    if flag in SEARCH_CLI_FLAGS:
        return {
            "flag": flag,
            "class": "applicable-search",
            "reason": "can alter capacity, scheduling, KV memory, or frontend throughput",
        }
    return {
        "flag": flag,
        "class": "review-required",
        "reason": "must be dispositioned against the exact stable image help",
    }


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    cli_args: tuple[str, ...]
    llm_options: dict[str, Any]
    hypothesis: str


def load_candidates(path: Path) -> list[Candidate]:
    rows = json.loads(path.read_text())["candidates"]
    candidates = [
        Candidate(
            name=row["name"],
            family=row["family"],
            cli_args=tuple(row["cli_args"]),
            llm_options=row["llm_options"],
            hypothesis=row["hypothesis"],
        )
        for row in rows
    ]
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")
    for candidate in candidates:
        overlap = FIXED_CLI_FLAGS.intersection(candidate.cli_args)
        if overlap:
            raise ValueError(
                f"{candidate.name} overrides fixed flags: {sorted(overlap)}"
            )
        if "max_seq_len" in candidate.llm_options:
            raise ValueError(f"{candidate.name} overrides fixed max_seq_len")
    return candidates


def select_image(*, stable_compatible: bool) -> str:
    return STABLE_IMAGE if stable_compatible else RC_FALLBACK_IMAGE


def container_name(candidate: Candidate) -> str:
    safe = re.sub(r"[^a-z0-9-]+", "-", candidate.name.lower()).strip("-")
    if not safe:
        raise ValueError("candidate name has no container-safe characters")
    return f"autoreply-m12-trtllm-{safe}"


def is_owned_container(name: str) -> bool:
    return bool(
        re.fullmatch(r"autoreply-m12-trtllm-[a-z0-9][a-z0-9-]*", name)
    )


def build_server_command(
    candidate: Candidate,
    *,
    image: str = STABLE_IMAGE,
    config_path: Path | None = None,
) -> list[str]:
    """Build a PyTorch-backend command with the exact served model identity."""
    command = [
        "docker",
        "run",
        "-d",
        "--init",
        "--name",
        container_name(candidate),
        "--gpus",
        "all",
        "--ipc=host",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "-e",
        "PYTHONPATH=/opt/autoreply",
        "-p",
        "8080:8000",
        "-v",
        f"{MODEL_PATH}:/models/{SERVED_MODEL}:ro",
        "-v",
        f"{WORKDIR / 'scripts/trtllm_autoreply_tokenizer.py'}:/opt/autoreply/trtllm_autoreply_tokenizer.py:ro",
        "-w",
        "/models",
    ]
    if config_path is not None:
        command.extend(["-v", f"{config_path}:/run/autoreply-options.yml:ro"])
    command.extend(
        [
            image,
            "trtllm-serve",
            "serve",
            SERVED_MODEL,
            "--backend",
            "pytorch",
            "--custom_module_dirs",
            "/opt/autoreply",
            "--custom_tokenizer",
            "trtllm_autoreply_tokenizer.AutoReplyTokenizer",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--max_batch_size",
            "96",
            "--max_num_tokens",
            "8192",
            "--max_seq_len",
            "8192",
            "--kv_cache_free_gpu_memory_fraction",
            "0.90",
        ]
    )
    if config_path is not None:
        command.extend(["--config", "/run/autoreply-options.yml"])
    command.extend(candidate.cli_args)
    return command
