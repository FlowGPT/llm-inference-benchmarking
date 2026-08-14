#!/usr/bin/env bash
# Run one production-aligned AutoReply QPS probe and retain all evidence.
set -euo pipefail

if [[ $# -gt 1 ]] || [[ $# -eq 0 && -z "${AUTOREPLY_QPS:-}" ]]; then
  echo "usage: $0 QPS (or set AUTOREPLY_QPS)" >&2
  exit 2
fi

QPS="${1:-${AUTOREPLY_QPS}}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ARTIFACT_DIR="${AUTOREPLY_ARTIFACT_DIR:-bench-runs/autoreply}"
ROUNDS="${AUTOREPLY_ROUNDS:-12}"
TAIL_WINDOW="${AUTOREPLY_TAIL_WINDOW:-6}"
API_BASE="${AUTOREPLY_API_BASE:-http://127.0.0.1:8080/v1}"
METRICS_URL="${AUTOREPLY_METRICS_URL:-http://127.0.0.1:8080/metrics}"
KEY="${AUTOREPLY_API_KEY:-autoreply-local-benchmark}"
mkdir -p "$ARTIFACT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE="${ARTIFACT_DIR}/qps_${QPS}_${STAMP}"
PREFIX_TOOL="/root/.cursor/skills/model-perf-binary-search/scripts/prefix_cache_hit_rate.py"
ANALYZER="/root/.cursor/skills/model-perf-binary-search/scripts/analyze_rounds.py"

printf '%s\n' "$BASE" > "${ARTIFACT_DIR}/.active_probe"
echo "PROBE_BASE=$BASE"
python3 "$PREFIX_TOOL" snapshot \
  --url "$METRICS_URL" --out "${BASE}.before.prom"

set +e
"$ROOT/.venv/bin/python" online_replay.py \
  --input datasets/autoreply_prod_dist_repeated_13x.jsonl \
  --preload-time 2 \
  --replay-mode qps --target-qps "$QPS" \
  --preselected-route \
  --serialize-conversations \
  --continuous-qps-window \
  --api-base "$API_BASE" \
  --api-key "$KEY" \
  --model kaonai/user-model-afs-sfw-m12-v7.1s1600-nvfp4 \
  --use-chat \
  --max-tokens 50 \
  --temperature 0.7 \
  --top-p 0.8 \
  --frequency-penalty 0.01 \
  --presence-penalty 0.01 \
  --disable-min-p \
  --extra-body-json '{"n":3,"stop":["<|im_end|>"],"top_k":-1}' \
  --round-duration 30 \
  --round-drain-timeout 300 \
  --request-timeout 600 \
  --max-rounds "$ROUNDS" \
  --e2e-slo 2.0 \
  --json-output "${BASE}.jsonl" \
  > "${BASE}.client.log" 2>&1
CLIENT_RC=$?
set -e
echo "$CLIENT_RC" > "${BASE}.client.rc"

python3 "$PREFIX_TOOL" snapshot \
  --url "$METRICS_URL" --out "${BASE}.after.prom"
set +e
python3 "$PREFIX_TOOL" diff \
  --before "${BASE}.before.prom" --after "${BASE}.after.prom" \
  | tee "${BASE}.prefix.json"
PREFIX_RC=${PIPESTATUS[0]}
python3 "$ANALYZER" \
  --json "${BASE}.jsonl" \
  --total-rounds "$ROUNDS" \
  --tail-window "$TAIL_WINDOW" \
  --slo 2.0 \
  --auto-steady \
  | tee "${BASE}.analysis.json"
ANALYZE_RC=${PIPESTATUS[0]}
set -e

printf '{"client_rc":%s,"prefix_rc":%s,"analyze_rc":%s}\n' \
  "$CLIENT_RC" "$PREFIX_RC" "$ANALYZE_RC" > "${BASE}.status.json"
rm -f "${ARTIFACT_DIR}/.active_probe"
echo "PROBE_DONE=$BASE"
exit "$ANALYZE_RC"
