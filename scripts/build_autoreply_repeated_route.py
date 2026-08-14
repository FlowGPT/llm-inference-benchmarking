#!/usr/bin/env python3
"""Repeat the verified 1000-row route for a continuous 12-round probe."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=13)
    parser.add_argument("--input", type=Path, default=Path("datasets/autoreply_prod_dist_1000.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("datasets/autoreply_prod_dist_repeated_13x.jsonl"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.open()]
    if len(rows) != 1000:
        raise ValueError(f"expected 1000 source rows, got {len(rows)}")
    with args.output.open("w") as output:
        for cycle in range(args.cycles):
            for index, row in enumerate(rows):
                minimal = {
                    "ts": 1755000000 + cycle * len(rows) + index,
                    "conv_id": f"{row['conv_id']}-cycle{cycle:02d}",
                    "body": row["body"],
                }
                output.write(json.dumps(minimal, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"source_rows": len(rows), "cycles": args.cycles, "output_rows": len(rows) * args.cycles, "output": str(args.output)}))


if __name__ == "__main__":
    main()
