#!/usr/bin/env bash
# Smoke-test the bundled helper scripts against known fixtures.
#
# Run after editing the bundled helpers to catch obvious regressions before
# pushing. Self-contained: no network, live engine, or GPU.
#
# Exit code: 0 if all checks pass, non-zero with case count otherwise.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIX="$HERE/fixtures"

PASS=0
FAIL=0
ok() { printf '[OK]   %s\n' "$*"; PASS=$((PASS + 1)); }
ko() { printf '[FAIL] %s\n' "$*" >&2; FAIL=$((FAIL + 1)); }

check() {
  local name="$1" expected_rc="$2" expected_grep="$3"; shift 3
  local out rc
  out="$("$@" 2>&1)"; rc=$?
  if [ "$rc" -eq "$expected_rc" ] && echo "$out" | grep -q -- "$expected_grep"; then
    ok "$name"
  else
    ko "$name -- rc=$rc (want $expected_rc), out=$out"
  fi
}

# analyze_rounds.py
check "analyze_rounds: PASS fixture -> status PASS, rc 0" 0 '"status": "PASS"' \
  python3 "$HERE/analyze_rounds.py" \
    --json "$FIX/analyze_pass.jsonl" \
    --total-rounds 12 --tail-window 6 --slo 6.5

check "analyze_rounds: FAIL fixture -> status FAIL, rc 1" 1 '"status": "FAIL"' \
  python3 "$HERE/analyze_rounds.py" \
    --json "$FIX/analyze_fail.jsonl" \
    --total-rounds 12 --tail-window 6 --slo 6.5

check "analyze_rounds: short jsonl -> NOT_ENOUGH_ROUNDS, rc 2" 2 '"status": "NOT_ENOUGH_ROUNDS"' \
  python3 "$HERE/analyze_rounds.py" \
    --json "$FIX/analyze_pass.jsonl" \
    --total-rounds 24 --tail-window 12 --slo 6.5

# analyze_rounds.py v2: --auto-steady + warmup_dominated

# Without --auto-steady, the warmup-dominated fixture FAILs (tail-6 = 9.17)
check "analyze_rounds: warmup-dominated fixture, no --auto-steady -> FAIL (tail)" 1 '"primary_metric": "tail"' \
  python3 "$HERE/analyze_rounds.py" \
    --json "$FIX/analyze_warmup_dominated_pass.jsonl" \
    --total-rounds 12 --tail-window 6 --slo 6.5

# With --auto-steady, the same fixture detects 5-round steady at 5.0s -> PASS
check "analyze_rounds: warmup-dominated fixture, --auto-steady -> PASS (steady)" 0 '"primary_metric": "steady"' \
  python3 "$HERE/analyze_rounds.py" \
    --json "$FIX/analyze_warmup_dominated_pass.jsonl" \
    --total-rounds 12 --tail-window 6 --slo 6.5 --auto-steady

check "analyze_rounds: warmup-dominated label triggers" 0 '"warmup_dominated": true' \
  python3 "$HERE/analyze_rounds.py" \
    --json "$FIX/analyze_warmup_dominated_pass.jsonl" \
    --total-rounds 12 --tail-window 6 --slo 6.5 --auto-steady

# Steady detected but avg over SLO -> still FAIL (algorithm is not a free pass)
check "analyze_rounds: steady-but-over-SLO fixture, --auto-steady -> FAIL" 1 '"status": "FAIL"' \
  python3 "$HERE/analyze_rounds.py" \
    --json "$FIX/analyze_steady_fail.jsonl" \
    --total-rounds 12 --tail-window 6 --slo 6.5 --auto-steady

# prefix_cache_hit_rate.py
check "prefix_cache: diff -> hit_rate 0.5, rc 0" 0 '"hit_rate": 0.5' \
  python3 "$HERE/prefix_cache_hit_rate.py" diff \
    --before "$FIX/prefix_before.prom" \
    --after "$FIX/prefix_after.prom"

check "prefix_cache: diff -> hits 100" 0 '"hits": 100' \
  python3 "$HERE/prefix_cache_hit_rate.py" diff \
    --before "$FIX/prefix_before.prom" \
    --after "$FIX/prefix_after.prom"

check "prefix_cache: no prefix metrics -> NO_PREFIX_METRICS, rc 2" 2 '"status": "NO_PREFIX_METRICS"' \
  python3 "$HERE/prefix_cache_hit_rate.py" diff \
    --before "$FIX/no_prefix_before.prom" \
    --after "$FIX/no_prefix_after.prom"

check "dataset: direct-path validation and failure gates" 0 'PASS' \
  bash "$HERE/test_prepare_dataset.sh"

check "repository: existing-checkout setup and safety gates" 0 'repository bootstrap rejects a dataset outside' \
  bash "$HERE/test_repo_mode.sh"

printf '\n=== %d passed, %d failed ===\n' "$PASS" "$FAIL"
exit "$FAIL"
