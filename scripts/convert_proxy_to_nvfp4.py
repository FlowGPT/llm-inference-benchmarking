#!/usr/bin/env python3
"""Convert a structural BF16 draft proxy to a serialized ModelOpt NVFP4 proxy.

This is a serving-cost proxy converter, not a quality-preserving PTQ tool.  It
replaces every two-dimensional linear weight with dense zero-valued packed
NVFP4 data and valid ModelOpt scales.  Dense NVFP4 kernels still execute their
normal work; placeholder output quality is intentionally out of scope.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


QUANTIZATION = {
    "quant_method": "modelopt",
    "quantization": {
        "quant_algo": "NVFP4",
        "kv_cache_quant_algo": "FP8",
        "group_size": 16,
        "exclude_modules": ["lm_head", "embed_tokens"],
    },
}


def convert(source: Path, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    tensors = load_file(source / "model.safetensors", device="cpu")
    converted: dict[str, torch.Tensor] = {}
    quantized = 0
    for name, tensor in tensors.items():
        is_embedding = name.endswith("markov_head.markov_w1.weight")
        if name.endswith(".weight") and tensor.ndim == 2 and not is_embedding:
            out_features, in_features = tensor.shape
            if in_features % 16:
                raise ValueError(f"{name}: input size {in_features} is not divisible by 16")
            prefix = name.removesuffix(".weight")
            converted[name] = torch.zeros(
                (out_features, in_features // 2), dtype=torch.uint8
            )
            converted[f"{prefix}.weight_scale"] = torch.ones(
                (out_features, in_features // 16), dtype=torch.float8_e4m3fn
            )
            converted[f"{prefix}.weight_scale_2"] = torch.tensor(
                1.0, dtype=torch.float32
            )
            converted[f"{prefix}.input_scale"] = torch.tensor(
                1.0, dtype=torch.float32
            )
            quantized += 1
        else:
            converted[name] = tensor.clone()

    save_file(converted, output / "model.safetensors")
    config = json.loads((source / "config.json").read_text())
    config["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "kv_cache_quant_algo": "FP8",
        "group_size": 16,
        "exclude_modules": ["lm_head", "embed_tokens"],
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "hf_quant_config.json").write_text(
        json.dumps(QUANTIZATION, indent=2) + "\n"
    )
    for filename in ("generation_config.json", "tokenizer_config.json"):
        candidate = source / filename
        if candidate.exists():
            shutil.copy2(candidate, output / filename)
    (output / "conversion.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "format": "ModelOpt NVFP4 structural cost proxy",
                "quantized_linear_weights": quantized,
                "quality_valid": False,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.source, args.output)


if __name__ == "__main__":
    main()
