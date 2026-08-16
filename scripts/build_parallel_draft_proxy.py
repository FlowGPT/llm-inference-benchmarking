#!/usr/bin/env python3
"""Build structurally valid proxy checkpoints for parallel draft benchmarks."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile


HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 14336
VOCAB_SIZE = 131072


@dataclass(frozen=True)
class ProxySpec:
    method: str
    config: dict[str, object]
    tensor_shapes: dict[str, tuple[int, ...]]
    num_speculative_tokens: int


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_dry_run(spec: ProxySpec, output: Path) -> None:
    """Write inspectable config and shape metadata without model weights."""
    tensors = [
        {
            "dtype": "BF16",
            "name": name,
            "nbytes": 2 * _numel(shape),
            "shape": list(shape),
            "source": "zero-init",
        }
        for name, shape in sorted(spec.tensor_shapes.items())
    ]
    _atomic_write_json(output / "config.json", spec.config)
    _atomic_write_json(
        output / "tensor-manifest.json",
        {"method": spec.method, "tensors": tensors},
    )


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dimension in shape:
        total *= dimension
    return total


def proxy_spec(method: str) -> ProxySpec:
    """Return the immutable structural specification for one proxy method."""
    if method == "eagle3":
        return ProxySpec(
            method="eagle3",
            config={
                "architectures": ["Eagle3LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": HIDDEN_SIZE,
                "intermediate_size": INTERMEDIATE_SIZE,
                "vocab_size": VOCAB_SIZE,
                "num_aux_hidden_states": 3,
                "eagle_config": {"use_aux_hidden_state": True},
            },
            tensor_shapes={"fc.weight": (HIDDEN_SIZE, 3 * HIDDEN_SIZE)},
            num_speculative_tokens=3,
        )
    if method == "dspark":
        return ProxySpec(
            method="dspark",
            config={
                "architectures": ["Qwen3DSparkModel"],
                "model_type": "qwen3",
                "hidden_size": HIDDEN_SIZE,
                "intermediate_size": INTERMEDIATE_SIZE,
                "vocab_size": VOCAB_SIZE,
                "dspark_block_size": 7,
                "n_predict": 7,
            },
            tensor_shapes={},
            num_speculative_tokens=7,
        )
    if method == "dflash":
        return ProxySpec(
            method="dflash",
            config={
                "architectures": ["DFlashDraftModel"],
                "model_type": "qwen3",
                "hidden_size": HIDDEN_SIZE,
                "intermediate_size": INTERMEDIATE_SIZE,
                "vocab_size": VOCAB_SIZE,
                "n_predict": 3,
                "dflash_config": {
                    "mask_token_id": VOCAB_SIZE - 1,
                    "target_layer_ids": [1, 20, 39],
                    "use_aux_hidden_state": True,
                    "causal": False,
                },
            },
            tensor_shapes={},
            num_speculative_tokens=3,
        )
    raise ValueError(f"unsupported proxy method: {method}")


def acceptance_schedules(k: int) -> dict[int, list[float]]:
    """Return approved unconditional per-position acceptance schedules."""
    if k == 3:
        return {
            50: [0.80, 0.50, 0.20],
            55: [0.85, 0.55, 0.25],
            60: [0.90, 0.60, 0.30],
        }
    if k == 7:
        return {
            50: [0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20],
            55: [0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25],
            60: [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30],
        }
    raise ValueError(f"unsupported speculative length: {k}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("eagle3", "dflash", "dspark"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        raise SystemExit("checkpoint generation is not implemented yet; use --dry-run")
    if args.method is None:
        raise SystemExit("--method is required with --dry-run")
    write_dry_run(proxy_spec(args.method), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
