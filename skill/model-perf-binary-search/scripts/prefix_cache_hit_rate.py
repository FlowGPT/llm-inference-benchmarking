#!/usr/bin/env python3
"""Snapshot a Prometheus /metrics endpoint and compute prefix-cache hit rate.

Two modes:
  snapshot  -- save current /metrics text to a file (header has ISO timestamp,
               source URL).
  diff      -- read two snapshot files, compute per-metric deltas, find the
               prefix-cache hit/query pair via a name heuristic, print one
               JSON line.

Framework-agnostic identification:
- Scan all metrics whose name (case-insensitive) contains "prefix".
- A "hit" metric matches /hit|match/i and does NOT contain "miss".
- A "query" metric matches /quer|lookup|total|request|access/i and does NOT
  contain "hit". (vLLM uses `*_queries_total`; some engines use `*_lookups`.)
- A "miss" metric is used as a fallback denominator: queries = hits + misses.
- If multiple candidates exist, pick the pair with the largest delta volume.

If no prefix-cache metrics are found, exit 2 and print the prefix metric
candidates seen, so the caller can fall back to log scraping or research the
engine's specific metric names.

Usage:
    # before a probe
    prefix_cache_hit_rate.py snapshot \\
        --url http://localhost:8100/metrics --out bench-runs/qps_5.0.before.prom

    # after a probe
    prefix_cache_hit_rate.py snapshot \\
        --url http://localhost:8100/metrics --out bench-runs/qps_5.0.after.prom

    # compute hit rate over the probe window
    prefix_cache_hit_rate.py diff \\
        --before bench-runs/qps_5.0.before.prom \\
        --after  bench-runs/qps_5.0.after.prom

JSON output keys (diff mode):
    status: OK | NO_PREFIX_METRICS | ERROR
    hit_rate: float in [0, 1] | null
    hits, queries: float (deltas over the probe window)
    hit_metric, denom_metric: chosen metric names
    denom_is_total: true if denominator is *_queries/lookups, false if hits+misses
    all_prefix_metrics_seen: list[str]
    note: human-readable explanation when status != OK

Exit codes: 0 OK, 2 NO_PREFIX_METRICS, 3 ERROR.
"""

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


_NAME_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")
_PREFIX_PAT = re.compile(r"prefix", re.IGNORECASE)
_HIT_PAT = re.compile(r"(hit|match)", re.IGNORECASE)
_QUERY_PAT = re.compile(r"(quer|lookup|total|request|access)", re.IGNORECASE)
_MISS_PAT = re.compile(r"miss", re.IGNORECASE)


def fetch_metrics(url: str, timeout: float = 10.0) -> str:
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_snapshot(text: str) -> dict[str, float]:
    """Parse Prometheus text format → {metric_name: sum_over_label_sets}."""
    totals: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _NAME_RE.match(line)
        if not m:
            continue
        name, _labels, value = m.group(1), m.group(2), m.group(3)
        try:
            v = float(value)
        except ValueError:
            continue
        totals[name] = totals.get(name, 0.0) + v
    return totals


def find_pair(deltas: dict[str, float]):
    """Pick (hit, denominator) pair among prefix metrics. Returns (vol, hit_key,
    denom_key, denom_is_total) or None."""
    prefix_keys = [k for k in deltas if _PREFIX_PAT.search(k)]
    hit_keys = [
        k for k in prefix_keys
        if _HIT_PAT.search(k) and not _MISS_PAT.search(k)
    ]
    query_keys = [
        k for k in prefix_keys
        if _QUERY_PAT.search(k) and not _HIT_PAT.search(k) and not _MISS_PAT.search(k)
    ]
    miss_keys = [k for k in prefix_keys if _MISS_PAT.search(k)]

    best = None  # (volume, hit_key, denom_key, denom_is_total)
    for h in hit_keys:
        for q in query_keys:
            vol = abs(deltas.get(h, 0.0)) + abs(deltas.get(q, 0.0))
            cand = (vol, h, q, True)
            if best is None or vol > best[0]:
                best = cand
        for mk in miss_keys:
            vol = abs(deltas.get(h, 0.0)) + abs(deltas.get(mk, 0.0))
            cand = (vol, h, mk, False)
            if best is None or vol > best[0]:
                best = cand
    return best, prefix_keys


def main_snapshot(args) -> int:
    try:
        text = fetch_metrics(args.url)
    except Exception as e:
        print(json.dumps({"status": "ERROR", "error": f"fetch: {e}"}))
        return 3
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# snapshot_iso={datetime.now(timezone.utc).isoformat()}\n"
        f"# snapshot_url={args.url}\n"
    )
    out.write_text(header + text)
    print(json.dumps({"status": "OK", "path": str(out), "bytes": len(text)}))
    return 0


def main_diff(args) -> int:
    try:
        before_text = Path(args.before).read_text()
        after_text = Path(args.after).read_text()
    except FileNotFoundError as e:
        print(json.dumps({"status": "ERROR", "error": str(e)}))
        return 3

    before = parse_snapshot(before_text)
    after = parse_snapshot(after_text)
    keys = set(before) | set(after)
    deltas = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in keys}

    best, prefix_keys = find_pair(deltas)
    if best is None:
        print(json.dumps({
            "status": "NO_PREFIX_METRICS",
            "note": (
                "No prefix-cache hit/query metrics found in /metrics. "
                "Fall back to log scraping (e.g. vLLM prints "
                "'Prefix cache hit rate: X.X%' periodically), or research "
                "this engine's specific metric names."
            ),
            "prefix_metric_candidates": sorted(prefix_keys),
        }))
        return 2

    _vol, hit_key, denom_key, denom_is_total = best
    hits = deltas[hit_key]
    denom_delta = deltas[denom_key]
    queries = denom_delta if denom_is_total else (hits + denom_delta)
    rate = (hits / queries) if queries > 0 else None
    out = {
        "status": "OK",
        "hit_rate": rate,
        "hits": hits,
        "queries": queries,
        "hit_metric": hit_key,
        "denom_metric": denom_key,
        "denom_is_total": denom_is_total,
        "all_prefix_metrics_seen": sorted(prefix_keys),
    }
    print(json.dumps(out))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("snapshot", help="scrape /metrics into a file")
    s.add_argument("--url", required=True, help="e.g. http://localhost:8100/metrics")
    s.add_argument("--out", required=True, help="snapshot file path (.prom)")

    d = sub.add_parser("diff", help="compute hit rate from two snapshots")
    d.add_argument("--before", required=True)
    d.add_argument("--after", required=True)

    args = p.parse_args()
    if args.cmd == "snapshot":
        return main_snapshot(args)
    return main_diff(args)


if __name__ == "__main__":
    sys.exit(main())
