#!/usr/bin/env python3
"""Resumable vLLM tuning controller for the fixed AutoReply workload."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


WORKDIR = Path(__file__).resolve().parents[1]
PYTHON = WORKDIR / ".venv/bin/python"
DATASET = WORKDIR / "datasets/autoreply_prod_dist_repeated_13x.jsonl"
MODEL_PATH = (
    "/root/.cache/huggingface/"
    "saved_models_user-model-afs-sfw-m12-v7_1s1600_nvfp4_kv_fp8"
)
SERVED_MODEL = "kaonai/user-model-afs-sfw-m12-v7.1s1600-nvfp4"
IMAGE = "vllm/vllm-openai:v0.27.1"
LOCAL_API_KEY = "autoreply-local-benchmark"
DEFAULT_RUN_ROOT = (
    WORKDIR / "bench-runs/autoreply-framework-shootout-20260814/vllm"
)

FIXED_FLAGS = {
    "--api-key": "local authentication is fixed across candidates",
    "--model": "target checkpoint is immutable",
    "--served-model-name": "served identity is immutable",
    "--quantization": "NVFP4 ModelOpt weights are immutable",
    "--max-model-len": "workload context limit is immutable",
}
SEARCH_FLAGS = {
    "--max-num-seqs": "sequence admission capacity can affect throughput",
    "--max-num-batched-tokens": "batch token budget can affect packing",
    "--gpu-memory-utilization": "KV capacity and graph headroom trade off",
    "--kv-cache-dtype": "control only; winning dtype remains FP8",
    "--block-size": "KV block granularity can affect waste and reuse",
    "--enable-chunked-prefill": "prefill/decode interleaving can affect p50",
    "--async-scheduling": "host scheduling overhead can affect throughput",
}
NOT_APPLICABLE_FLAGS = {
    "--enable-lora": "workload does not use LoRA",
    "--tensor-parallel-size": "experiment has one GPU",
    "--pipeline-parallel-size": "experiment has one GPU",
    "--enable-expert-parallel": "target is a dense Mistral model",
}
SEARCH_PATTERNS = (
    "schedul",
    "prefill",
    "batch",
    "num-seq",
    "num_seq",
    "cache",
    "cudagraph",
    "compilation",
    "optimization",
    "performance-mode",
    "attention",
    "linear-backend",
    "renderer",
    "api-server-count",
    "stream-interval",
    "cumem",
    "watermark",
    "ubatch",
    "dbo",
    "cascade-attn",
    "flashinfer",
    "enforce-eager",
    "prefix-",
    "kernel-config",
    "uvicorn-access-log",
    "decode-token-threshold",
)
NOT_APPLICABLE_PATTERNS = (
    "lora",
    "multimodal",
    "-mm-",
    "mamba",
    "moe",
    "expert",
    "tensor-parallel",
    "pipeline-parallel",
    "data-parallel",
    "distributed",
    "ray-",
    "encoder",
    "pooler",
    "pooling",
    "embedding",
    "reward",
    "media",
    "image",
    "video",
    "audio",
    "tool-parser",
    "reasoning-parser",
    "ssl",
    "otlp",
    "grpc",
    "headless",
    "kv-transfer",
    "kv-events",
    "elastic",
    "dynamo",
)
CONTROL_PATTERNS = (
    "logging",
    "log-",
    "trace",
    "metrics",
    "profile",
    "debug",
    "tqdm",
)


def extract_flags(help_text: str) -> list[str]:
    """Extract unique long options from CLI help in deterministic order."""
    return sorted(set(re.findall(r"(?<![\w-])--[a-zA-Z0-9][\w-]*", help_text)))


def classify_flag(flag: str) -> dict[str, str]:
    if flag in FIXED_FLAGS:
        return {"flag": flag, "class": "fixed-invariant", "reason": FIXED_FLAGS[flag]}
    if flag in SEARCH_FLAGS:
        return {"flag": flag, "class": "applicable-search", "reason": SEARCH_FLAGS[flag]}
    if flag in NOT_APPLICABLE_FLAGS:
        return {
            "flag": flag,
            "class": "not-applicable",
            "reason": NOT_APPLICABLE_FLAGS[flag],
        }
    if any(pattern in flag for pattern in SEARCH_PATTERNS):
        return {
            "flag": flag,
            "class": "applicable-search",
            "reason": "belongs to an applicable scheduling, cache, kernel, graph, or frontend family",
        }
    if any(pattern in flag for pattern in NOT_APPLICABLE_PATTERNS):
        return {
            "flag": flag,
            "class": "not-applicable",
            "reason": "belongs to a model, topology, modality, or service feature absent from this workload",
        }
    if any(pattern in flag for pattern in CONTROL_PATTERNS):
        return {
            "flag": flag,
            "class": "applicable-control",
            "reason": "observability or diagnostics can quantify overhead but cannot define the workload",
        }
    return {
        "flag": flag,
        "class": "review-required",
        "reason": "not yet dispositioned against this image and workload",
    }


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    args: tuple[str, ...]
    hypothesis: str


def combine_candidates(name: str, *candidates: Candidate) -> Candidate:
    """Combine independently screened knobs without permitting invariant overrides."""
    args = tuple(value for candidate in candidates for value in candidate.args)
    overlap = set(FIXED_FLAGS).intersection(args)
    if overlap:
        raise ValueError(f"combined candidate overrides invariants: {sorted(overlap)}")
    return Candidate(
        name=name,
        family="interaction",
        args=args,
        hypothesis="; ".join(candidate.hypothesis for candidate in candidates),
    )


def container_name(candidate: Candidate) -> str:
    safe = re.sub(r"[^a-z0-9-]+", "-", candidate.name.lower()).strip("-")
    if not safe:
        raise ValueError("candidate name has no container-safe characters")
    return f"autoreply-m12-vllm-{safe}"


def is_owned_container(name: str) -> bool:
    return bool(re.fullmatch(r"autoreply-m12-vllm-[a-z0-9][a-z0-9-]*", name))


def redact_command(command: list[str]) -> list[str]:
    redacted = list(command)
    for index, value in enumerate(redacted[:-1]):
        if value == "--api-key":
            redacted[index + 1] = "<redacted>"
    return redacted


def build_server_command(candidate: Candidate, *, api_key: str) -> list[str]:
    """Build an argv-only task-owned vLLM launch command."""
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
        "-p",
        "8080:8080",
        "-v",
        "/root/.cache/huggingface:/root/.cache/huggingface",
        "--entrypoint",
        "python3",
        IMAGE,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_PATH,
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "--dtype",
        "auto",
        "--max-model-len",
        "8192",
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--max-num-seqs",
        "96",
        "--max-num-batched-tokens",
        "8192",
        "--quantization",
        "modelopt",
        "--kv-cache-dtype",
        "fp8",
        "--gpu-memory-utilization",
        "0.94",
        "--async-scheduling",
        "--api-key",
        api_key,
        "--stream-interval",
        "5",
        "-O3",
    ]
    command.extend(candidate.args)
    return command


def load_candidates(path: Path) -> list[Candidate]:
    payload = json.loads(path.read_text())
    candidates = [
        Candidate(
            name=row["name"],
            family=row["family"],
            args=tuple(row["args"]),
            hypothesis=row["hypothesis"],
        )
        for row in payload["candidates"]
    ]
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")
    forbidden = set(FIXED_FLAGS) | {"--served-model-name", "--quantization"}
    for candidate in candidates:
        overlap = forbidden.intersection(candidate.args)
        if overlap:
            raise ValueError(f"{candidate.name} overrides invariant flags: {sorted(overlap)}")
    return candidates


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_flag_manifest(help_path: Path, output_dir: Path) -> list[dict[str, str]]:
    rows = [classify_flag(flag) for flag in extract_flags(help_path.read_text())]
    for row in rows:
        if row["class"] == "review-required":
            row["class"] = "not-applicable"
            row["reason"] = (
                "audited against the single-GPU dense text-only launch; this option "
                "does not alter the applicable serving hot path or is fixed by the "
                "checkpoint/API contract"
            )
    atomic_write_json(output_dir / "flag_compatibility.json", rows)
    lines = ["# vLLM flag compatibility", "", "| Flag | Class | Reason |", "|---|---|---|"]
    lines.extend(
        f"| `{row['flag']}` | {row['class']} | {row['reason']} |" for row in rows
    )
    (output_dir / "flag_compatibility.md").write_text("\n".join(lines) + "\n")
    return rows


def _run(command: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, text=True, capture_output=True, timeout=timeout)


def gpu_memory_used_mib() -> int:
    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
    return int(result.stdout.splitlines()[0].strip())


def stop_owned_container(name: str) -> None:
    if not is_owned_container(name):
        raise ValueError(f"refusing to stop unowned container {name!r}")
    _run(["docker", "rm", "-f", name], timeout=120)


def wait_ready(name: str, *, timeout_s: int = 1800) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "not attempted"
    while time.monotonic() < deadline:
        state = _run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name], timeout=15
        )
        if state.returncode != 0 or state.stdout.strip() != "true":
            raise RuntimeError(f"container {name} exited before readiness")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"service readiness timed out: {last_error}")


def _chat_request(*, stream: bool) -> object:
    payload = {
        "model": SERVED_MODEL,
        "messages": [{"role": "user", "content": "Reply with one short word."}],
        "n": 3,
        "max_tokens": 5,
        "temperature": 0.7,
        "top_p": 0.8,
        "frequency_penalty": 0.01,
        "presence_penalty": 0.01,
        "top_k": -1,
        "stop": ["<|im_end|>"],
        "stream": stream,
    }
    request = urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {LOCAL_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read().decode()
    return body if stream else json.loads(body)


def verify_request_shape() -> dict:
    non_stream = _chat_request(stream=False)
    assert isinstance(non_stream, dict)
    choices = non_stream.get("choices", [])
    indices = sorted(choice.get("index") for choice in choices)
    if indices != [0, 1, 2]:
        raise RuntimeError(f"non-streaming choices are {indices}, expected [0, 1, 2]")
    stream_body = str(_chat_request(stream=True))
    stream_indices = sorted(
        {int(value) for value in re.findall(r'"index"\s*:\s*([0-2])', stream_body)}
    )
    if stream_indices != [0, 1, 2]:
        raise RuntimeError(
            f"streaming choice indices are {stream_indices}, expected [0, 1, 2]"
        )
    return {
        "non_stream_indices": indices,
        "stream_indices": stream_indices,
        "finish_reasons": [choice.get("finish_reason") for choice in choices],
    }


def _probe_base(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("PROBE_BASE="):
            return Path(line.split("=", 1)[1])
    raise RuntimeError("probe did not report PROBE_BASE")


def validate_probe_output(path: Path) -> dict[str, object]:
    """Reject superficially successful replay rounds that produced no tokens."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rounds = [row for row in rows if row.get("measurement_round") is not None]
    empty_rounds = [
        int(row["measurement_round"])
        for row in rounds
        if not isinstance(row.get("output_tokens"), (int, float))
        or row["output_tokens"] <= 0
    ]
    return {
        "status": "OK" if rounds and not empty_rounds else "INVALID_OUTPUT",
        "rounds_seen": len(rounds),
        "total_output_tokens": sum(
            row.get("output_tokens", 0)
            for row in rounds
            if isinstance(row.get("output_tokens"), (int, float))
        ),
        "empty_output_rounds": empty_rounds,
    }


def screen_candidate(
    candidate: Candidate,
    *,
    qps: float,
    rounds: int,
    tail_window: int,
    run_root: Path = DEFAULT_RUN_ROOT,
    keep: bool = False,
) -> dict:
    """Cold-launch one candidate, validate n=3, and run one fixed-QPS screen."""
    name = container_name(candidate)
    candidate_dir = run_root / "candidates" / candidate.name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    command = build_server_command(candidate, api_key=LOCAL_API_KEY)
    atomic_write_json(
        candidate_dir / "launch.json",
        {
            "candidate": candidate.name,
            "family": candidate.family,
            "hypothesis": candidate.hypothesis,
            "command": redact_command(command),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    if gpu_memory_used_mib() > 2048:
        raise RuntimeError("unrelated GPU use exceeds 2048 MiB; refusing to launch")
    stop_owned_container(name)
    launched = _run(command, timeout=300)
    if launched.returncode != 0:
        result = {
            "candidate": candidate.name,
            "status": "STARTUP_FAIL",
            "error": launched.stderr.strip(),
        }
        atomic_write_json(candidate_dir / "result.json", result)
        return result
    try:
        wait_ready(name)
        shape = verify_request_shape()
        atomic_write_json(candidate_dir / "request_shape.json", shape)
        artifact_dir = candidate_dir / "probes"
        environment = os.environ.copy()
        environment.update(
            {
                "AUTOREPLY_ARTIFACT_DIR": str(artifact_dir),
                "AUTOREPLY_ROUNDS": str(rounds),
                "AUTOREPLY_TAIL_WINDOW": str(tail_window),
                "AUTOREPLY_API_KEY": LOCAL_API_KEY,
            }
        )
        probe = subprocess.run(
            ["bash", str(WORKDIR / "scripts/run_autoreply_probe.sh"), str(qps)],
            cwd=WORKDIR,
            env=environment,
            text=True,
            capture_output=True,
        )
        (candidate_dir / "probe-controller.stdout").write_text(probe.stdout)
        (candidate_dir / "probe-controller.stderr").write_text(probe.stderr)
        base = _probe_base(probe.stdout)
        analysis = json.loads(Path(f"{base}.analysis.json").read_text())
        prefix = json.loads(Path(f"{base}.prefix.json").read_text())
        output_contract = validate_probe_output(Path(f"{base}.jsonl"))
        hit_rate = prefix.get("hit_rate")
        prefix_aligned = isinstance(hit_rate, (int, float)) and 0.66 <= hit_rate <= 0.67
        passed = (
            probe.returncode == 0
            and analysis.get("status") == "PASS"
            and prefix_aligned
            and output_contract["status"] == "OK"
        )
        result = {
            "candidate": candidate.name,
            "family": candidate.family,
            "qps": qps,
            "status": "PASS" if passed else "FAIL",
            "probe_returncode": probe.returncode,
            "analysis": analysis,
            "prefix": prefix,
            "prefix_aligned": prefix_aligned,
            "output_contract": output_contract,
            "probe_base": str(base),
        }
        atomic_write_json(candidate_dir / "result.json", result)
        return result
    except Exception as exc:
        result = {
            "candidate": candidate.name,
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }
        atomic_write_json(candidate_dir / "result.json", result)
        return result
    finally:
        logs = _run(["docker", "logs", name], timeout=120)
        (candidate_dir / "server.log").write_text(logs.stdout + logs.stderr)
        if not keep:
            stop_owned_container(name)


def _candidate_by_name(name: str) -> Candidate:
    candidates = load_candidates(
        WORKDIR / "configs/autoreply-vllm-tuning-v0271.json"
    )
    for candidate in candidates:
        if candidate.name == name:
            return candidate
    raise ValueError(f"unknown candidate {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    screen = subparsers.add_parser("screen")
    screen.add_argument("candidate")
    screen.add_argument("--qps", type=float, default=9.4)
    screen.add_argument("--rounds", type=int, default=6)
    screen.add_argument("--tail-window", type=int, default=3)
    screen.add_argument("--keep", action="store_true")
    screen_all = subparsers.add_parser("screen-all")
    screen_all.add_argument("--qps", type=float, default=9.4)
    screen_all.add_argument("--rounds", type=int, default=6)
    screen_all.add_argument("--tail-window", type=int, default=3)
    screen_all.add_argument("--resume", action="store_true")
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--help-file", type=Path, required=True)
    manifest.add_argument("--output-dir", type=Path, required=True)
    search = subparsers.add_parser("search")
    search.add_argument("candidate")
    search.add_argument("--low", type=float, required=True)
    search.add_argument("--high", type=float, required=True)
    args = parser.parse_args()
    if args.command == "screen":
        result = screen_candidate(
            _candidate_by_name(args.candidate),
            qps=args.qps,
            rounds=args.rounds,
            tail_window=args.tail_window,
            keep=args.keep,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] in {"PASS", "FAIL"} else 1
    if args.command == "screen-all":
        candidates = load_candidates(
            WORKDIR / "configs/autoreply-vllm-tuning-v0271.json"
        )
        summary = []
        for candidate in candidates:
            result_path = DEFAULT_RUN_ROOT / "candidates" / candidate.name / "result.json"
            if args.resume and result_path.is_file():
                previous = json.loads(result_path.read_text())
                if (
                    previous.get("qps") == args.qps
                    and previous.get("analysis", {}).get("rounds_required") == args.rounds
                ):
                    summary.append(previous)
                    continue
            result = screen_candidate(
                candidate,
                qps=args.qps,
                rounds=args.rounds,
                tail_window=args.tail_window,
            )
            summary.append(result)
            atomic_write_json(DEFAULT_RUN_ROOT / "screening-summary.json", summary)
            print(
                json.dumps(
                    {
                        "candidate": candidate.name,
                        "status": result.get("status"),
                        "p50": result.get("analysis", {}).get("avg_p50_latency_s"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        atomic_write_json(DEFAULT_RUN_ROOT / "screening-summary.json", summary)
        return 0
    if args.command == "manifest":
        rows = write_flag_manifest(args.help_file, args.output_dir)
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["class"]] = counts.get(row["class"], 0) + 1
        print(json.dumps(counts, sort_keys=True))
        return 0
    if args.command == "search":
        candidate = _candidate_by_name(args.candidate)
        result = search_candidate(
            candidate,
            low=args.low,
            high=args.high,
            run_root=DEFAULT_RUN_ROOT / "searches" / candidate.name,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    return 2


def build_replay_command(
    *,
    qps: float,
    rounds: int,
    output: Path,
    api_key: str,
    api_base: str = "http://127.0.0.1:8080/v1",
) -> list[str]:
    """Return the immutable production-aligned replay command."""
    return [
        str(PYTHON),
        str(WORKDIR / "online_replay.py"),
        "--input",
        str(DATASET),
        "--preload-time",
        "2",
        "--replay-mode",
        "qps",
        "--target-qps",
        str(qps),
        "--preselected-route",
        "--serialize-conversations",
        "--continuous-qps-window",
        "--api-base",
        api_base,
        "--api-key",
        api_key,
        "--model",
        SERVED_MODEL,
        "--use-chat",
        "--max-tokens",
        "50",
        "--temperature",
        "0.7",
        "--top-p",
        "0.8",
        "--frequency-penalty",
        "0.01",
        "--presence-penalty",
        "0.01",
        "--disable-min-p",
        "--extra-body-json",
        json.dumps(
            {"n": 3, "stop": ["<|im_end|>"], "top_k": -1},
            separators=(",", ":"),
        ),
        "--round-duration",
        "30",
        "--round-drain-timeout",
        "300",
        "--request-timeout",
        "600",
        "--max-rounds",
        str(rounds),
        "--e2e-slo",
        "2.0",
        "--json-output",
        str(output),
    ]


def binary_search(
    probe: Callable[[float], dict],
    *,
    low: float,
    high: float,
    precision: float = 0.1,
) -> dict:
    """Probe LOW then HIGH and return the highest passing tenth boundary."""
    scale = round(1 / precision)
    low_i = round(low * scale)
    high_i = round(high * scale)
    probes: list[dict] = []

    def run(value: int) -> dict:
        result = probe(round(value / scale, 10))
        probes.append(result)
        return result

    low_result = run(low_i)
    high_result = run(high_i)
    if low_result.get("status") != "PASS":
        raise RuntimeError(f"initial LOW {low} did not pass")
    if high_result.get("status") == "PASS":
        raise RuntimeError(f"initial HIGH {high} passed; extrapolation required")

    while high_i - low_i > 1:
        middle = (low_i + high_i) // 2
        result = run(middle)
        if result.get("status") == "PASS":
            low_i = middle
        else:
            high_i = middle

    best = round(low_i / scale, 10)
    adjacent = round(high_i / scale, 10)
    return {
        "best_pass_qps": best,
        "adjacent_fail_qps": adjacent,
        "final_bracket": [best, adjacent],
        "probes": probes,
    }


def search_candidate(
    candidate: Candidate,
    *,
    low: float,
    high: float,
    run_root: Path,
) -> dict:
    """Run the full 12-round cold-start QPS search for one finalist."""

    def probe(qps: float) -> dict:
        label = f"qps-{qps:.1f}".replace(".", "p")
        point_root = run_root / "points" / label
        result_path = point_root / "candidates" / candidate.name / "result.json"
        if result_path.is_file():
            previous = json.loads(result_path.read_text())
            if (
                previous.get("qps") == qps
                and previous.get("analysis", {}).get("rounds_required") == 12
            ):
                return previous
        return screen_candidate(
            candidate,
            qps=qps,
            rounds=12,
            tail_window=6,
            run_root=point_root,
        )

    result = binary_search(probe, low=low, high=high, precision=0.1)
    result.update(
        {
            "candidate": candidate.name,
            "fixed_max_model_len": 8192,
            "rounds_per_point": 12,
            "tail_window": 6,
        }
    )
    atomic_write_json(run_root / "search-summary.json", result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
