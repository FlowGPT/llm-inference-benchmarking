#!/usr/bin/env python3
"""Assert AutoReply production and boundary dataset invariants."""

import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from build_autoreply_prod_datasets import MODEL, QUOTAS, bucket_for, token_count


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "datasets/autoreply_prod_dist_1000.jsonl"
BOUNDARY = ROOT / "datasets/autoreply_prod_boundary_300.jsonl"
REPORT = ROOT / "datasets/autoreply_prod_verify.json"
EXPECTED_CATEGORIES = {
    "trim_edge_3743",
    "fixed_prompt_4k_6k",
    "fixed_prompt_6k_8k",
    "extreme_8k",
    "long_personality_short_history",
    "history_50",
    "history_51_v4_async",
    "consecutive_same_role",
    "long_single_message",
    "multilingual",
    "entry_direct",
    "entry_v4_async",
}


def read_jsonl(path):
    with path.open() as source:
        return [json.loads(line) for line in source]


def verify_common(tokenizer, records):
    ids = [x["conv_id"] for x in records]
    assert len(ids) == len(set(ids)), "conv_id values must be unique"
    counts = []
    for item in records:
        body = item["body"]
        messages = body["prompt"]
        assert messages and messages[0]["role"] == "system"
        assert all(set(m) == {"role", "content"} for m in messages)
        assert all(messages[i]["role"] != messages[i - 1]["role"] for i in range(2, len(messages)))
        assert messages[0]["content"].startswith(item["assembly"]["system_base"])
        assert body["max_tokens"] == 50
        assert body["temperature"] == 0.7 and body["top_p"] == 0.8
        assert body["frequency_penalty"] == 0.01 and body["presence_penalty"] == 0.01
        assert "top_k" not in body and "min_p" not in body
        assert item["model"]["expected_n"] == 3 and item["model"]["parse_mode"] == "n_choices"
        count = token_count(tokenizer, messages)
        assert count == item["model"]["prompt_tokens_chat_template"]
        assert bucket_for(count) == item["model"]["bucket"]
        counts.append(count)
    return counts


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    dist = read_jsonl(DIST)
    boundary = read_jsonl(BOUNDARY)
    assert len(dist) == 1000
    dist_counts = verify_common(tokenizer, dist)
    observed = Counter(x["model"]["bucket"] for x in dist)
    assert observed == Counter(QUOTAS), (observed, QUOTAS)
    max_bucket_run = 1
    current_run = 1
    for previous, current in zip(dist, dist[1:]):
        if previous["model"]["bucket"] == current["model"]["bucket"]:
            current_run += 1
            max_bucket_run = max(max_bucket_run, current_run)
        else:
            current_run = 1
    assert max_bucket_run <= 10, "bucket-grouped ordering distorts continuous windows"
    assert all(x["assembly"]["prefix_reuse_source"] == "n=3 shared prefill only" for x in dist)
    unique_prompts = {json.dumps(x["body"]["prompt"], ensure_ascii=False, sort_keys=True) for x in dist}
    assert len(unique_prompts) == len(dist), "cross-request duplicate prompts inflate prefix hits"
    first_blocks = {
        tuple(tokenizer.apply_chat_template(x["body"]["prompt"], tokenize=True, add_generation_prompt=True)["input_ids"][:16])
        for x in dist
    }
    assert len(first_blocks) == len(dist), "cross-request first-block reuse inflates prefix hits"
    expected_reuse = 2 / 3
    assert all(x["assembly"]["turns_out"] <= (51 if x["assembly"]["entry"] == "v4_async" else 50) for x in dist)

    assert len(boundary) >= 300
    boundary_counts = verify_common(tokenizer, boundary)
    categories = Counter(c for x in boundary for c in x["assembly"]["boundary_categories"])
    assert EXPECTED_CATEGORIES <= set(categories)
    assert all(n < 8192 for n in boundary_counts)
    assert any(x["assembly"]["trimmed"] and 3720 <= x["assembly"]["local_tokens_after_trim"] <= 3743 for x in boundary)
    assert any(
        "long_personality_short_history" in x["assembly"]["boundary_categories"]
        and x["assembly"]["turns_out"] <= 2
        and x["model"]["prompt_tokens_chat_template"] > 4096
        for x in boundary
    )
    assert any(6000 <= n < 8000 for n in boundary_counts)
    assert any(8100 <= n <= 8142 for n in boundary_counts)
    assert any(x["assembly"]["turns_in"] == 50 for x in boundary)
    assert any(x["assembly"]["turns_in"] == 51 for x in boundary)

    report = {
        "status": "PASS",
        "dist_rows": len(dist),
        "dist_buckets": dict(observed),
        "dist_token_min": min(dist_counts),
        "dist_token_max": max(dist_counts),
        "dist_max_same_bucket_run": max_bucket_run,
        "dist_unique_prompts": len(unique_prompts),
        "dist_unique_first_blocks": len(first_blocks),
        "expected_n3_shared_prefill_reuse": expected_reuse,
        "boundary_rows": len(boundary),
        "boundary_categories": dict(categories),
        "boundary_token_min": min(boundary_counts),
        "boundary_token_max": max(boundary_counts),
        "checks": [
            "unique conv_id and valid OpenAI messages",
            "system/personality prefix preserved; history-only trimming",
            "production sampling fields exact; no top_k/min_p in log body",
            "chat-template token counts and production quotas exact",
            "all required boundary categories present and under max-model-len",
        ],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
