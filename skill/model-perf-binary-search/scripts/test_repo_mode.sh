#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEST_ROOT"' EXIT

make_repo() {
  local repo_root="$1"
  mkdir -p "$repo_root/skill/model-perf-binary-search/scripts"
  cp "$SCRIPT_DIR/bootstrap.sh" "$repo_root/skill/model-perf-binary-search/scripts/bootstrap.sh"
  cp "$SCRIPT_DIR/prepare_dataset.sh" "$repo_root/skill/model-perf-binary-search/scripts/prepare_dataset.sh"
  cp "$SCRIPT_DIR/health_check.py" "$repo_root/skill/model-perf-binary-search/scripts/health_check.py"
  printf '%s\n' 'openai>=1.0.0' > "$repo_root/requirements.txt"
  printf '%s\n' '"--serialize-conversations"' '"--continuous-qps-window"' '"--preselected-route"' > "$repo_root/online_replay.py"
  git -C "$repo_root" init -q
  git -C "$repo_root" add requirements.txt online_replay.py
  git -C "$repo_root" -c user.name=test -c user.email=test@example.invalid commit -qm init
}

run_check() {
  local repo_root="$1"
  local dataset="$2"
  LLM_BENCH_DIR="$TEST_ROOT/ignored-workdir" LLM_BENCH_SHARED_MOUNT="$SHARED_ROOT" LLM_BENCH_DATASET_SRC="$dataset" bash "$repo_root/skill/model-perf-binary-search/scripts/bootstrap.sh" --check-only
}

SHARED_ROOT="$TEST_ROOT/shared"
DATA_DIR="$SHARED_ROOT/data"
mkdir -p "$DATA_DIR"
printf '%s\n' '{"id":"one"}' > "$DATA_DIR/input.jsonl"
printf '%s\n' '{"id":"outside"}' > "$TEST_ROOT/outside.jsonl"

GOOD_REPO="$TEST_ROOT/good"
make_repo "$GOOD_REPO"
printf '%s\n' dirty >> "$GOOD_REPO/online_replay.py"
BEFORE_HEAD="$(git -C "$GOOD_REPO" rev-parse HEAD)"
BEFORE_BRANCH="$(git -C "$GOOD_REPO" branch --show-current)"
BEFORE_STATUS="$(git -C "$GOOD_REPO" status --porcelain)"

OUTPUT="$(
  run_check "$GOOD_REPO" "$DATA_DIR/input.jsonl"
)"
eval "$OUTPUT"

test "$REPO_ROOT" = "$GOOD_REPO"
test "$DATASET" = "$DATA_DIR/input.jsonl"
test "$PYTHON" = "$GOOD_REPO/.venv/bin/python"
test "$(git -C "$GOOD_REPO" rev-parse HEAD)" = "$BEFORE_HEAD"
test "$(git -C "$GOOD_REPO" branch --show-current)" = "$BEFORE_BRANCH"
test "$(git -C "$GOOD_REPO" status --porcelain)" = "$BEFORE_STATUS"

echo "[OK] repository bootstrap discovers a dirty existing checkout safely"

SETUP_REPO="$TEST_ROOT/setup"
make_repo "$SETUP_REPO"
FAKE_BIN="$TEST_ROOT/bin"
UV_LOG="$TEST_ROOT/uv.log"
PYTHON_LOG="$TEST_ROOT/python.log"
mkdir -p "$FAKE_BIN"
FAKE_PYTHON_STUB="$FAKE_BIN/python-stub"
printf '%s\n' '#!/usr/bin/env bash' 'printf "%s\\n" "$*" >> "$PYTHON_LOG"' > "$FAKE_PYTHON_STUB"
chmod +x "$FAKE_PYTHON_STUB"
printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail' 'printf "%s\\n" "$*" >> "$UV_LOG"' 'if [ "${1:-}" = "venv" ]; then' '  mkdir -p .venv/bin' '  cp "$FAKE_PYTHON_STUB" .venv/bin/python' '  chmod +x .venv/bin/python' 'fi' > "$FAKE_BIN/uv"
chmod +x "$FAKE_BIN/uv"
export UV_LOG PYTHON_LOG FAKE_PYTHON_STUB

PATH="$FAKE_BIN:$PATH" LLM_BENCH_DIR="$SETUP_REPO" LLM_BENCH_SHARED_MOUNT="$SHARED_ROOT" LLM_BENCH_DATASET_SRC="$DATA_DIR/input.jsonl" bash "$SETUP_REPO/skill/model-perf-binary-search/scripts/bootstrap.sh" >/dev/null

test -x "$SETUP_REPO/.venv/bin/python"
test -d "$SETUP_REPO/bench-runs"
grep -Fxq "venv" "$UV_LOG"
grep -Fxq "pip install -r requirements.txt" "$UV_LOG"
grep -Fxq "pip install requests pytest" "$UV_LOG"
grep -Fq "health_check.py --workdir $SETUP_REPO" "$PYTHON_LOG"

echo "[OK] repository bootstrap prepares the local environment"

BAD_REPO="$TEST_ROOT/bad"
make_repo "$BAD_REPO"
sed -i '/preselected-route/d' "$BAD_REPO/online_replay.py"
if run_check "$BAD_REPO" "$DATA_DIR/input.jsonl" >/dev/null 2>&1; then
  echo "expected missing replay capability to fail" >&2
  exit 1
fi

echo "[OK] repository bootstrap rejects a missing replay capability"

if grep -Eq 'git[[:space:]]+(clone|fetch|checkout|pull)' "$SCRIPT_DIR/bootstrap.sh"; then
  echo "bootstrap must not mutate Git state" >&2
  exit 1
fi

echo "[OK] repository bootstrap contains no Git mutation command"

if run_check "$GOOD_REPO" "$TEST_ROOT/outside.jsonl" >/dev/null 2>&1; then
  echo "expected outside dataset to fail" >&2
  exit 1
fi

echo "[OK] repository bootstrap rejects a dataset outside the shared directory"
