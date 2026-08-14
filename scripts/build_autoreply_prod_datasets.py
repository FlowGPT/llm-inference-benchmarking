#!/usr/bin/env python3
"""Build production-shaped AutoReply replay and boundary datasets."""

import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/auto_reply_pairs_filtered_ppl_tagged.jsonl"
DIST = ROOT / "datasets/autoreply_prod_dist_1000.jsonl"
BOUNDARY = ROOT / "datasets/autoreply_prod_boundary_300.jsonl"
MODEL = Path("/root/.cache/huggingface/saved_models_user-model-afs-sfw-m12-v7_1s1600_nvfp4_kv_fp8")
TRIGGER = "Suggest three short in-character next-user-turn replies."
LOCAL_BUDGET = 3743
QUOTAS = {
    "<1k": 37,
    "1k-2k": 108,
    "2k-3k": 108,
    "3k-3.5k": 77,
    "3.5k-3.8k": 283,
    "3.8k-4k": 373,
    "4k-4095": 9,
    "4096-8k": 5,
}
TARGETS = {
    "<1k": 850,
    "1k-2k": 1500,
    "2k-3k": 2500,
    "3k-3.5k": 3250,
    "3.5k-3.8k": 3740,
    "3.8k-4k": 3900,
    "4k-4095": 4050,
    "4096-8k": 6000,
}


def token_count(tokenizer, messages):
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return len(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)


def merge_roles(messages):
    merged = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "")
        if role not in {"user", "assistant"}:
            continue
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n" + content
        else:
            merged.append({"role": role, "content": content})
    return merged


def system_prompt(row):
    return (
        f"Character profile:\n{row.get('personality') or ''}\n\n"
        f"Scenario:\n{row.get('scenario') or ''}\n\n"
        f"Opening context:\n{row.get('welcome') or ''}\n\n"
        "Generate short, natural next-user-turn suggestions for this in-character chat. "
        "Respect the profile, scenario, opening context, conversation language, and recent history. "
        "Keep suggestions distinct and concise; do not narrate or explain."
    )


def pad_system_to(tokenizer, messages, target, current=None):
    """Append neutral system context until the rendered prompt is at target."""
    current = token_count(tokenizer, messages) if current is None else current
    if current >= target:
        return current, False
    original = messages[0]["content"]
    repeats = target - current
    messages[0]["content"] = original + " context" * repeats
    count = token_count(tokenizer, messages)
    return count, True


def assemble(tokenizer, row, target, entry="direct", history_override=None):
    base_system = system_prompt(row)
    history_raw = [dict(m) for m in (history_override if history_override is not None else row.get("history", [])[-50:])]
    if entry == "v4_async":
        suggestion = next((str(x) for x in row.get("suggestions", []) if x), "...")
        history_raw.append({"role": "assistant", "content": suggestion})
    history = merge_roles(history_raw)
    trimmed = False

    def rendered():
        return [{"role": "system", "content": base_system}] + history + [{"role": "user", "content": TRIGGER}]

    trim_limit = min(target, LOCAL_BUDGET)
    if history_override is None:
        # Cheap coarse trim before rendering potentially enormous source histories.
        char_budget = max(0, trim_limit * 3 - len(base_system) - len(TRIGGER))
        kept_chars = 0
        keep_from = len(history)
        for i in range(len(history) - 1, -1, -1):
            size = len(history[i]["content"])
            if kept_chars + size > char_budget:
                break
            kept_chars += size
            keep_from = i
        if keep_from:
            history = history[keep_from:]
            trimmed = True
    messages = rendered()
    initial_count = token_count(tokenizer, messages)
    if history and initial_count > trim_limit:
        original_history = history
        lo, hi = 1, len(original_history)
        while lo < hi:
            mid = (lo + hi) // 2
            history = merge_roles(original_history[mid:])
            messages = rendered()
            if token_count(tokenizer, messages) <= trim_limit:
                hi = mid
            else:
                lo = mid + 1
        history = merge_roles(original_history[lo:])
        messages = rendered()
        trimmed = True
    local_count = token_count(tokenizer, messages)
    if local_count > target:
        return None
    final_count, padded = pad_system_to(tokenizer, messages, target, local_count)
    return {
        "messages": messages,
        "history_raw": history_raw,
        "history_after_merge": history,
        "trimmed": trimmed,
        "local_count": local_count,
        "final_count": final_count,
        "padded": padded,
        "base_system": base_system,
    }


def bucket_for(n):
    if n < 1000:
        return "<1k"
    if n < 2000:
        return "1k-2k"
    if n < 3000:
        return "2k-3k"
    if n < 3500:
        return "3k-3.5k"
    if n < 3800:
        return "3.5k-3.8k"
    if n < 4000:
        return "3.8k-4k"
    if n <= 4095:
        return "4k-4095"
    if n < 8000:
        return "4096-8k"
    return ">=8k"


def record(row, built, conv_id, bucket, entry, categories=None):
    language = row.get("language") or "unknown"
    return {
        "ts": int(row.get("conv_create_at") or 1755000000),
        "conv_id": conv_id,
        "assembly": {
            "source_row": row.get("id") or row.get("conversation_id"),
            "personality": row.get("personality") or "",
            "scenario": row.get("scenario") or "",
            "welcome": row.get("welcome") or "",
            "system_base": built["base_system"],
            "history_raw": built["history_raw"],
            "history_after_merge": built["history_after_merge"],
            "entry": entry,
            "language": language,
            "last_translated": language.lower().startswith("en"),
            "pm_note": "approx-271+28, not production PM dump",
            "local_budget": LOCAL_BUDGET,
            "local_tokens_after_trim": built["local_count"],
            "trimmed": built["trimmed"],
            "turns_in": len(built["history_raw"]),
            "turns_out": len(built["history_after_merge"]),
            "padded_to_hit_provider_bucket": built["padded"],
            "boundary_categories": categories or [],
        },
        "model": {
            "prompt_tokens_chat_template": built["final_count"],
            "bucket": bucket,
            "expected_n": 3,
            "parse_mode": "n_choices",
        },
        "body": {
            "prompt": built["messages"],
            "max_tokens": 50,
            "temperature": 0.7,
            "top_p": 0.8,
            "frequency_penalty": 0.01,
            "presence_penalty": 0.01,
        },
    }


def load_rows(limit=2200, stride=100):
    rows = []
    with SOURCE.open() as source:
        for line_number, line in enumerate(source):
            if line_number % stride:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def build_dist(tokenizer, rows):
    output = []
    seen_prompts = set()
    seen_first_blocks = set()
    cursor = 0
    for bucket, quota in QUOTAS.items():
        made = 0
        attempts = 0
        while made < quota:
            row = rows[cursor % len(rows)]
            cursor += 1
            attempts += 1
            entry = "v4_async" if (cursor % 5 == 0) else "direct"
            built = assemble(tokenizer, row, TARGETS[bucket], entry)
            if built is None or bucket_for(built["final_count"]) != bucket:
                if attempts > len(rows) * 3:
                    raise RuntimeError(f"could not fill {bucket}: {made}/{quota}")
                continue
            prompt_key = json.dumps(built["messages"], ensure_ascii=False, sort_keys=True)
            if prompt_key in seen_prompts:
                continue
            encoded = tokenizer.apply_chat_template(built["messages"], tokenize=True, add_generation_prompt=True)
            first_block = tuple(encoded["input_ids"][:16])
            if first_block in seen_first_blocks:
                continue
            seen_prompts.add(prompt_key)
            seen_first_blocks.add(first_block)
            item = record(row, built, f"ar-dist-{len(output) + 1:04d}", bucket, entry)
            item["ts"] = 1755000000 + len(output)
            item["assembly"]["prefix_reuse_source"] = "n=3 shared prefill only"
            output.append(item)
            made += 1
    random.Random(20260813).shuffle(output)
    for index, item in enumerate(output):
        item["ts"] = 1755000000 + index
    return output


def short_history(turns):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Short conversation turn {i + 1}."}
        for i in range(turns)
    ]


def boundary_specs():
    specs = []
    specs += [("trim_edge_3743", 3743, "direct", None)] * 40
    specs += [("fixed_prompt_4k_6k", 5000, "direct", short_history(2))] * 35
    specs += [("fixed_prompt_6k_8k", 7000, "direct", short_history(2))] * 35
    specs += [("extreme_8k", 8120, "direct", short_history(2))] * 30
    specs += [("long_personality_short_history", 4600, "direct", short_history(2))] * 30
    specs += [("history_50", 3500, "direct", short_history(50))] * 30
    specs += [("history_51_v4_async", 3600, "v4_async", short_history(50))] * 30
    repeated = short_history(10)
    repeated.insert(1, {"role": "user", "content": "Second consecutive user message for merge verification."})
    specs += [("consecutive_same_role", 3000, "direct", repeated)] * 25
    long_one = [{"role": "user", "content": "multisentence " * 2200}, {"role": "assistant", "content": "I understand."}]
    specs += [("long_single_message", 3740, "direct", long_one)] * 25
    specs += [("multilingual", 3300, "direct", None)] * 30
    return specs


def build_boundary(tokenizer, rows):
    multilingual = [r for r in rows if not str(r.get("language") or "").lower().startswith("en")]
    output = []
    for i, (category, target, entry, history) in enumerate(boundary_specs()):
        pool = multilingual if category == "multilingual" else rows
        row = None
        built = None
        if category == "trim_edge_3743":
            for offset in range(min(20, len(pool))):
                candidate = pool[(i + offset) % len(pool)]
                for words in range(20, 101):
                    fine_history = [
                        {
                            "role": "user" if turn % 2 == 0 else "assistant",
                            "content": (f"detail{turn} " * words).strip(),
                        }
                        for turn in range(50)
                    ]
                    candidate_built = assemble(tokenizer, candidate, target, entry, fine_history)
                    if candidate_built and candidate_built["trimmed"] and 3720 <= candidate_built["local_count"] <= 3743:
                        row, built = candidate, candidate_built
                        break
                if built is not None:
                    break
        else:
            for offset in range(len(pool)):
                candidate = pool[(i + offset) % len(pool)]
                candidate_built = assemble(tokenizer, candidate, target, entry, history)
                if candidate_built is not None:
                    row, built = candidate, candidate_built
                    break
        if row is None or built is None:
            raise RuntimeError(f"could not build boundary sample {category} at {target}")
        categories = [category, f"entry_{entry}"]
        output.append(record(row, built, f"ar-boundary-{i + 1:04d}", bucket_for(built["final_count"]), entry, categories))
    return output


def write_jsonl(path, records):
    with path.open("w") as output:
        for item in records:
            output.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    rows = load_rows()
    dist = build_dist(tokenizer, rows)
    boundary = build_boundary(tokenizer, rows)
    write_jsonl(DIST, dist)
    write_jsonl(BOUNDARY, boundary)
    print(json.dumps({
        "dist": len(dist),
        "dist_buckets": Counter(x["model"]["bucket"] for x in dist),
        "boundary": len(boundary),
        "boundary_categories": Counter(c for x in boundary for c in x["assembly"]["boundary_categories"]),
    }, ensure_ascii=False, default=dict, indent=2))


if __name__ == "__main__":
    main()
