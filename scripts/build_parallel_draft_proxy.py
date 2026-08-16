#!/usr/bin/env python3
"""Build structurally valid proxy checkpoints for parallel draft benchmarks."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import struct
import tempfile


HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 14336
VOCAB_SIZE = 131072
NUM_DRAFT_LAYERS = 5
MARKOV_RANK = 256
DTYPE_NBYTES = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4}


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


def read_safetensors_header(path: Path) -> dict[str, dict[str, object]]:
    """Read safetensors metadata without materializing tensor payloads."""
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"invalid safetensors header prefix: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        header = json.loads(stream.read(header_length))
    return {name: value for name, value in header.items() if name != "__metadata__"}


def build_manifest(spec: ProxySpec, donor_dir: Path) -> dict[str, object]:
    """Describe every tensor needed to materialize one proxy checkpoint."""
    donor_header = read_safetensors_header(donor_dir / "model.safetensors")
    tensors: list[dict[str, object]] = []
    donor_layer = {
        name: metadata
        for name, metadata in donor_header.items()
        if name.startswith("layers.0.")
    }
    layer_count = 1 if spec.method == "eagle3" else NUM_DRAFT_LAYERS
    for layer_idx in range(layer_count):
        for source_name, metadata in sorted(donor_layer.items()):
            if spec.method == "eagle3" and ".self_attn." in source_name and any(
                projection in source_name
                for projection in ("q_proj", "k_proj", "v_proj")
            ):
                continue
            name = source_name.replace("layers.0.", f"layers.{layer_idx}.", 1)
            shape = tuple(int(value) for value in metadata["shape"])
            dtype = str(metadata["dtype"])
            tensors.append(
                {
                    "dtype": dtype,
                    "name": name,
                    "nbytes": DTYPE_NBYTES[dtype] * _numel(shape),
                    "shape": list(shape),
                    "source": "classic-eagle-donor",
                    "source_name": source_name,
                }
            )

    method_shapes = dict(spec.tensor_shapes)
    if spec.method in {"dflash", "dspark"}:
        for layer_idx in range(NUM_DRAFT_LAYERS):
            for suffix, shape in (
                ("input_layernorm.weight", (HIDDEN_SIZE,)),
                ("self_attn.q_norm.weight", (128,)),
                ("self_attn.k_norm.weight", (128,)),
            ):
                method_shapes[f"layers.{layer_idx}.{suffix}"] = shape

    for name, shape in sorted(method_shapes.items()):
        tensors.append(
            {
                "dtype": "BF16",
                "name": name,
                "nbytes": DTYPE_NBYTES["BF16"] * _numel(shape),
                "shape": list(shape),
                "source": (
                    "identity-init" if name.endswith("norm.weight") else "zero-init"
                ),
            }
        )
    return {
        "method": spec.method,
        "tensors": tensors,
        "total_nbytes": sum(int(tensor["nbytes"]) for tensor in tensors),
    }


def materialize_checkpoint(
    spec: ProxySpec, donor_dir: Path, output: Path, *, force: bool = False
) -> None:
    """Materialize one proxy checkpoint using the exact runtime dependencies."""
    final_weights = output / "model.safetensors"
    if final_weights.exists() and final_weights.stat().st_size > 0 and not force:
        raise FileExistsError(f"refusing to overwrite checkpoint: {final_weights}")

    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError(
            "checkpoint materialization requires torch and safetensors; "
            "run this command inside the vLLM image"
        ) from error

    manifest = build_manifest(spec, donor_dir)
    donor_path = donor_dir / "model.safetensors"
    tensors = {}
    with safe_open(donor_path, framework="pt", device="cpu") as donor:
        for tensor in manifest["tensors"]:
            name = str(tensor["name"])
            shape = tuple(int(value) for value in tensor["shape"])
            if tensor["source"] == "classic-eagle-donor":
                source_name = str(tensor.get("source_name", name))
                # Repeated draft layers intentionally start from the same donor
                # values, but safetensors requires independent storage per key.
                value = donor.get_tensor(source_name).clone()
                if tuple(value.shape) != shape:
                    raise ValueError(
                        f"donor shape mismatch for {name}: {tuple(value.shape)} != {shape}"
                    )
                tensors[name] = value
            elif tensor["source"] == "zero-init":
                tensors[name] = torch.zeros(shape, dtype=torch.bfloat16)
            elif tensor["source"] == "identity-init":
                tensors[name] = torch.ones(shape, dtype=torch.bfloat16)
            else:
                raise ValueError(f"unsupported tensor source: {tensor['source']}")

    donor_config = json.loads((donor_dir / "config.json").read_text())
    config = {**donor_config, **spec.config}
    output.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".model.safetensors.", suffix=".tmp", dir=output
    )
    os.close(fd)
    try:
        save_file(tensors, temporary, metadata={"format": "pt"})
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, final_weights)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    _atomic_write_json(output / "config.json", config)
    _atomic_write_json(output / "tensor-manifest.json", manifest)


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
            tensor_shapes={
                "fc.weight": (HIDDEN_SIZE, 3 * HIDDEN_SIZE),
                "layers.0.input_layernorm.weight": (HIDDEN_SIZE,),
                "layers.0.hidden_norm.weight": (HIDDEN_SIZE,),
                "layers.0.self_attn.q_proj.weight": (4096, 2 * HIDDEN_SIZE),
                "layers.0.self_attn.k_proj.weight": (1024, 2 * HIDDEN_SIZE),
                "layers.0.self_attn.v_proj.weight": (1024, 2 * HIDDEN_SIZE),
                "norm.weight": (HIDDEN_SIZE,),
            },
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
                "num_hidden_layers": NUM_DRAFT_LAYERS,
                "num_target_layers": 40,
                "block_size": 7,
                "dspark_block_size": 7,
                "n_predict": 7,
                "markov_rank": MARKOV_RANK,
                "layer_types": ["full_attention"] * NUM_DRAFT_LAYERS,
                "dflash_config": {
                    "mask_token_id": VOCAB_SIZE - 1,
                    "target_layer_ids": [1, 10, 20, 30, 39],
                    "use_aux_hidden_state": True,
                    "causal": False,
                },
            },
            tensor_shapes={
                "fc.weight": (HIDDEN_SIZE, 5 * HIDDEN_SIZE),
                "hidden_norm.weight": (HIDDEN_SIZE,),
                "norm.weight": (HIDDEN_SIZE,),
                "markov_head.markov_w1.weight": (VOCAB_SIZE, MARKOV_RANK),
                "markov_head.markov_w2.weight": (VOCAB_SIZE, MARKOV_RANK),
            },
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
                "num_hidden_layers": NUM_DRAFT_LAYERS,
                "num_target_layers": 40,
                "block_size": 3,
                "n_predict": 3,
                "layer_types": ["full_attention"] * NUM_DRAFT_LAYERS,
                "dflash_config": {
                    "mask_token_id": VOCAB_SIZE - 1,
                    "target_layer_ids": [1, 20, 39],
                    "use_aux_hidden_state": True,
                    "causal": False,
                },
            },
            tensor_shapes={
                "fc.weight": (HIDDEN_SIZE, 3 * HIDDEN_SIZE),
                "hidden_norm.weight": (HIDDEN_SIZE,),
                "norm.weight": (HIDDEN_SIZE,),
            },
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
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--donor", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.all == (args.method is not None):
        raise SystemExit("choose exactly one of --method or --all")
    methods = ("eagle3", "dflash", "dspark") if args.all else (args.method,)
    assert all(method is not None for method in methods)
    for method in methods:
        spec = proxy_spec(str(method))
        output = args.output / spec.method if args.all else args.output
        if args.dry_run:
            write_dry_run(spec, output)
            continue
        if args.donor is None:
            raise SystemExit("--donor is required for checkpoint generation")
        materialize_checkpoint(spec, args.donor, output, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
