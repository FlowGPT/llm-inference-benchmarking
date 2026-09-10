#!/usr/bin/env python3
"""Analyze online_replay.py JSON output and decide if a QPS run met the SLO.

Reads a JSON-lines file produced by `online_replay.py --json-output`, where
each line is one round's metrics. Computes the average of the per-round
`Latency.p50` over the last N rounds and compares it against an SLO.

NEW (v2): --auto-steady flag enables steady-state-window detection.
  Walks backward from the last round, including a round in the steady window
  if it's within `--steady-tolerance` (default 0.30, i.e. ±30%) of the median
  of already-included rounds. Stops at the first round too far off. If the
  detected window has at least `--steady-min-window` (default 3) rounds, that
  window's avg becomes the PRIMARY PASS/FAIL signal (instead of the fixed
  tail window). If detection fails (fewer than min_window consistent rounds at
  the end), falls back to the original tail-window behavior with a warning.

Also (always emitted, even without --auto-steady): a warmup_dominated label.
  TRUE when tail_N / tail_3 > 1.5 OR tail_N / last_round > 2.0. Signals that
  the tail window still contains warmup backlog and the verdict may be unfair.

Exit codes:
  0 -> PASS  (avg over primary window < SLO AND we have enough rounds)
  1 -> FAIL  (avg over primary window >= SLO with enough rounds)
  2 -> NOT_ENOUGH_ROUNDS (json contains fewer rounds than required)
  3 -> usage / parse error

Stdout (always, single line of JSON):
  {"status": "PASS"|"FAIL"|"NOT_ENOUGH_ROUNDS",
   "rounds_seen": int, "rounds_required": int,
   "tail_window": int,
   "avg_p50_latency_s": float | null,        # the primary metric (steady if found, else tail)
   "primary_metric": "steady"|"tail",
   "slo_s": float,
   "per_round_p50": [float, ...],
   "tail_avg_p50_latency_s": float | null,   # always the tail-N avg, for backward compat
   "steady_window_start_idx": int | null,    # 0-indexed, null if not detected
   "steady_window_size": int,                # 0 if not detected
   "steady_avg_p50_latency_s": float | null,
   "warmup_dominated": bool,
   "warmup_ratio_tail_over_tail3": float | null,
   "notes": [str, ...]}
"""

import argparse
import json
import statistics
import sys
from pathlib import Path


def find_steady_window(p50s, tolerance=0.30, min_window=3):
    """Walk backward from end; include round in window if within ±tolerance of
    the running median. Stop at first round too far off. Return (start_idx,
    window_size, window_values) or (None, 0, [])."""
    if not p50s:
        return None, 0, []
    window = [p50s[-1]]
    for i in range(len(p50s) - 2, -1, -1):
        med = statistics.median(window)
        if med <= 0:
            break
        if abs(p50s[i] - med) / med <= tolerance:
            window.insert(0, p50s[i])
        else:
            break
    if len(window) >= min_window:
        return len(p50s) - len(window), len(window), window
    return None, 0, []


def is_warmup_dominated(per_round_p50, tail_window):
    """TRUE when tail_N avg / tail_3 avg > 1.5 OR tail_N / last_round > 2.0.
    Signals that the tail window still contains warmup backlog."""
    if len(per_round_p50) < tail_window:
        return False, None
    tail = per_round_p50[-tail_window:]
    tail_avg = sum(tail) / len(tail)
    last_3_avg = sum(per_round_p50[-3:]) / 3 if len(per_round_p50) >= 3 else None
    last_round = per_round_p50[-1]
    ratio_t_t3 = (tail_avg / last_3_avg) if last_3_avg and last_3_avg > 0 else None
    flag = False
    if ratio_t_t3 is not None and ratio_t_t3 > 1.5:
        flag = True
    if last_round > 0 and tail_avg / last_round > 2.0:
        flag = True
    return flag, ratio_t_t3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, nargs="+",
                        help="One or more JSON-lines files from --json-output. "
                             "When multiple are given (sharded clients), per-round "
                             "p50 is averaged across files for the same round index.")
    parser.add_argument("--total-rounds", type=int, required=True,
                        help="Total rounds expected (e.g. 8 or 16).")
    parser.add_argument("--tail-window", type=int, required=True,
                        help="Number of trailing rounds to average (e.g. 4 or 8).")
    parser.add_argument("--slo", type=float, default=6.5,
                        help="E2E p50 latency SLO in seconds (default 6.5).")
    parser.add_argument("--auto-steady", action="store_true",
                        help="Use auto-detected steady window as primary PASS/FAIL "
                             "metric when one is found; otherwise fall back to tail "
                             "window. Without this flag, behavior is identical to v1.")
    parser.add_argument("--steady-tolerance", type=float, default=0.30,
                        help="Per-round deviation tolerance for steady-window "
                             "detection (default 0.30 = ±30%% of running median).")
    parser.add_argument("--steady-min-window", type=int, default=3,
                        help="Minimum number of rounds for steady-window detection "
                             "to succeed (default 3).")
    args = parser.parse_args()

    per_file_rounds: list[list[float]] = []
    for path_str in args.json:
        path = Path(path_str)
        if not path.exists():
            print(json.dumps({"status": "ERROR", "error": f"missing file {path_str}"}))
            return 3
        rounds: list[float] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lat = rec.get("Latency", {})
                p50 = lat.get("p50")
                if p50 is None:
                    continue
                rounds.append(float(p50))
        per_file_rounds.append(rounds)

    rounds_seen = min(len(r) for r in per_file_rounds) if per_file_rounds else 0
    per_round_p50 = []
    for i in range(rounds_seen):
        per_round_p50.append(sum(r[i] for r in per_file_rounds) / len(per_file_rounds))

    result = {
        "status": "NOT_ENOUGH_ROUNDS",
        "rounds_seen": rounds_seen,
        "rounds_required": args.total_rounds,
        "tail_window": args.tail_window,
        "avg_p50_latency_s": None,
        "primary_metric": None,
        "slo_s": args.slo,
        "per_round_p50": per_round_p50,
        "tail_avg_p50_latency_s": None,
        "steady_window_start_idx": None,
        "steady_window_size": 0,
        "steady_avg_p50_latency_s": None,
        "warmup_dominated": False,
        "warmup_ratio_tail_over_tail3": None,
        "notes": [],
    }

    if rounds_seen < args.total_rounds:
        print(json.dumps(result))
        return 2

    tail = per_round_p50[-args.tail_window:]
    tail_avg = sum(tail) / len(tail)
    result["tail_avg_p50_latency_s"] = tail_avg

    # warmup-dominated label (always computed)
    wd, ratio = is_warmup_dominated(per_round_p50, args.tail_window)
    result["warmup_dominated"] = wd
    result["warmup_ratio_tail_over_tail3"] = ratio
    if wd:
        result["notes"].append(
            f"warmup-dominated: tail-{args.tail_window} avg {tail_avg:.2f}s vs "
            f"tail-3 avg {sum(per_round_p50[-3:])/3:.2f}s "
            f"(ratio {ratio:.2f}); the SLO verdict may be unfair to the engine."
        )

    # steady-window detection (always computed when we have enough rounds)
    s_start, s_size, s_window = find_steady_window(
        per_round_p50, tolerance=args.steady_tolerance, min_window=args.steady_min_window
    )
    if s_size > 0:
        steady_avg = sum(s_window) / s_size
        result["steady_window_start_idx"] = s_start
        result["steady_window_size"] = s_size
        result["steady_avg_p50_latency_s"] = steady_avg

    # decide primary metric
    if args.auto_steady and s_size > 0:
        primary = "steady"
        avg = steady_avg
        result["notes"].append(
            f"auto-steady: detected {s_size}-round window starting at round "
            f"{s_start + 1} (1-indexed), avg {steady_avg:.2f}s used as primary."
        )
    else:
        primary = "tail"
        avg = tail_avg
        if args.auto_steady and s_size == 0:
            result["notes"].append(
                f"auto-steady: no steady window detected within --steady-tolerance "
                f"{args.steady_tolerance:.2f} and --steady-min-window "
                f"{args.steady_min_window}; falling back to tail-{args.tail_window}."
            )

    result["primary_metric"] = primary
    result["avg_p50_latency_s"] = avg
    result["status"] = "PASS" if avg < args.slo else "FAIL"
    print(json.dumps(result))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
