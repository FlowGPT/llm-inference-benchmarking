#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[bootstrap] ERROR: %s\n' "$*" >&2
  exit 1
}

CHECK_ONLY=0
if [ "${1:-}" = "--check-only" ]; then
  CHECK_ONLY=1
  shift
fi
[ "$#" -eq 0 ] || fail "usage: bootstrap.sh [--check-only]"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SHARED_MOUNT="${LLM_BENCH_SHARED_MOUNT:-/mnt/shared/sss}"
SHARED_DATA_DIR="$SHARED_MOUNT/data"
DATASET_SRC="${LLM_BENCH_DATASET_SRC:-}"

[ -e "$REPO_ROOT/.git" ] || fail "技能必须位于已拉取的 Git 仓库中：$REPO_ROOT"
[ -f "$REPO_ROOT/online_replay.py" ] || fail "仓库缺少 online_replay.py：$REPO_ROOT"
[ -f "$REPO_ROOT/requirements.txt" ] || fail "仓库缺少 requirements.txt：$REPO_ROOT"

for flag in serialize-conversations continuous-qps-window preselected-route; do
  grep -Eq -- "['\"]--$flag['\"]" "$REPO_ROOT/online_replay.py" ||
    fail "当前 checkout 的 online_replay.py 缺少 --$flag；请使用包含回放因果功能的版本。"
done

DATASET_ASSIGNMENT="$(
  bash "$SCRIPT_DIR/prepare_dataset.sh" "$SHARED_DATA_DIR" "$DATASET_SRC"
)" || fail "数据集验证失败"
eval "$DATASET_ASSIGNMENT"

PYTHON="$REPO_ROOT/.venv/bin/python"

print_assignments() {
  printf 'REPO_ROOT=%q\n' "$REPO_ROOT"
  printf 'DATASET=%q\n' "$DATASET"
  printf 'PYTHON=%q\n' "$PYTHON"
}

if [ "$CHECK_ONLY" -eq 1 ]; then
  print_assignments
  exit 0
fi

command -v uv >/dev/null 2>&1 ||
  fail "缺少 uv；请先安装 uv 后重试。"

cd "$REPO_ROOT"
if [ ! -d .venv ]; then
  uv venv
fi
uv pip install -r requirements.txt
uv pip install requests pytest

[ -x "$PYTHON" ] || fail "虚拟环境缺少可执行 Python：$PYTHON"
mkdir -p bench-runs

set +e
"$PYTHON" "$SCRIPT_DIR/health_check.py" --workdir "$REPO_ROOT" >&2
HEALTH_RC=$?
set -e
printf '[bootstrap] health check exit=%s (json: %s/.health_check.json)\n' "$HEALTH_RC" "$REPO_ROOT" >&2

print_assignments
