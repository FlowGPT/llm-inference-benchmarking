# Repository-Native Model Performance Skill Implementation Plan

> **For agents:** Use `executing-plans` inline. Steps use `- [ ]`
> checkboxes. Do not commit or push because the repository already contains
> unrelated staged and unstaged user changes.

**Goal:** Bundle a repository-native `model-perf-binary-search` skill that
operates on an already-cloned checkout without using the global Cursor skill or
changing Git state.

**Architecture:** Copy the proven deterministic helpers into
`skill/model-perf-binary-search`, replace the clone-oriented bootstrap with a
repository-discovery bootstrap, and keep normal instructions concise by moving
conditional service and tuning guidance into references. Capability checks
against the checked-out `online_replay.py` replace branch selection.

**Reference implementations:** The source behavior is
`/root/.cursor/skills/model-perf-binary-search`; the approved design is
`docs/specs/2026-09-09-repository-native-model-perf-skill-design.md`.

---

## File Map

- **Create:** `skill/model-perf-binary-search/SKILL.md` — repo-native entry
  workflow, required inputs, probe algorithm, and reporting contract.
- **Create:** `skill/model-perf-binary-search/references/service-lifecycle.md`
  — process/Docker lifecycle, readiness, health gates, and progress reporting.
- **Create:** `skill/model-perf-binary-search/references/tuning.md` —
  optional generic and feature-enablement tuning.
- **Copy unchanged:** `skill/model-perf-binary-search/scripts/analyze_rounds.py`
  — PASS/FAIL steady-window analyzer.
- **Copy then replace:** `skill/model-perf-binary-search/scripts/bootstrap.sh`
  — preserve a runnable baseline first, then discover and validate the existing
  repository and prepare its environment.
- **Copy unchanged:** `skill/model-perf-binary-search/scripts/health_check.py`
  — hardware and disk preflight.
- **Copy unchanged:**
  `skill/model-perf-binary-search/scripts/prefix_cache_hit_rate.py` —
  framework-neutral prefix-cache metrics.
- **Copy then simplify:**
  `skill/model-perf-binary-search/scripts/prepare_dataset.sh` — validate an
  explicit shared dataset and print its canonical path without copying it.
- **Copy then update:**
  `skill/model-perf-binary-search/scripts/test_prepare_dataset.sh` — verify
  direct-path validation, empty/missing inputs, and directory boundaries.
- **Copy and update:** `skill/model-perf-binary-search/scripts/smoke.sh` —
  retain helper checks and invoke repository-mode tests.
- **Create:**
  `skill/model-perf-binary-search/scripts/test_repo_mode.sh` — isolated
  bootstrap and Git-safety integration tests.
- **Copy unchanged:** `skill/model-perf-binary-search/scripts/fixtures/*` —
  analyzer and metrics fixtures.
- **Create:** `.gitignore` — ignore only generated environment, cache, health,
  and benchmark-output paths.

## Task 1: Copy the Proven Helper Baseline

**Files:**

- Create `skill/model-perf-binary-search/scripts/analyze_rounds.py`
- Create `skill/model-perf-binary-search/scripts/bootstrap.sh`
- Create `skill/model-perf-binary-search/scripts/health_check.py`
- Create `skill/model-perf-binary-search/scripts/prefix_cache_hit_rate.py`
- Create `skill/model-perf-binary-search/scripts/prepare_dataset.sh`
- Create `skill/model-perf-binary-search/scripts/test_prepare_dataset.sh`
- Create `skill/model-perf-binary-search/scripts/smoke.sh`
- Create `skill/model-perf-binary-search/scripts/fixtures/*`

**Steps:**

- [x] Copy only the helper files required by the approved design:

```bash
mkdir -p skill/model-perf-binary-search/scripts
cp -a /root/.cursor/skills/model-perf-binary-search/scripts/bootstrap.sh \
  /root/.cursor/skills/model-perf-binary-search/scripts/analyze_rounds.py \
  /root/.cursor/skills/model-perf-binary-search/scripts/health_check.py \
  /root/.cursor/skills/model-perf-binary-search/scripts/prefix_cache_hit_rate.py \
  /root/.cursor/skills/model-perf-binary-search/scripts/prepare_dataset.sh \
  /root/.cursor/skills/model-perf-binary-search/scripts/test_prepare_dataset.sh \
  /root/.cursor/skills/model-perf-binary-search/scripts/smoke.sh \
  skill/model-perf-binary-search/scripts/
cp -a /root/.cursor/skills/model-perf-binary-search/scripts/fixtures \
  skill/model-perf-binary-search/scripts/
```

- [x] Verify copied helpers are byte-identical:

```bash
for file in bootstrap.sh analyze_rounds.py health_check.py prefix_cache_hit_rate.py \
  prepare_dataset.sh test_prepare_dataset.sh smoke.sh; do
  cmp "/root/.cursor/skills/model-perf-binary-search/scripts/$file" \
      "skill/model-perf-binary-search/scripts/$file"
done
diff -qr /root/.cursor/skills/model-perf-binary-search/scripts/fixtures \
  skill/model-perf-binary-search/scripts/fixtures
```

Expected: exit 0 and no output.

- [x] Run the copied baseline smoke suite:

```bash
bash skill/model-perf-binary-search/scripts/smoke.sh
```

Expected: `=== 11 passed, 0 failed ===`.

## Task 2: Add a Failing Repository-Mode Test

**Files:**

- Create `skill/model-perf-binary-search/scripts/test_repo_mode.sh`

**Steps:**

- [x] Add an isolated shell test that creates temporary compatible and
  incompatible repositories, invokes `bootstrap.sh --check-only`, and checks
  Git state before and after:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

make_repo() {
  local root="$1"
  mkdir -p "$root/skill/model-perf-binary-search/scripts"
  cp "$SCRIPT_DIR/bootstrap.sh" \
    "$root/skill/model-perf-binary-search/scripts/bootstrap.sh"
  cp "$SCRIPT_DIR/prepare_dataset.sh" \
    "$root/skill/model-perf-binary-search/scripts/prepare_dataset.sh"
  printf '%s\n' 'openai>=1.0.0' > "$root/requirements.txt"
  printf '%s\n' \
    '"--serialize-conversations"' \
    '"--continuous-qps-window"' \
    '"--preselected-route"' > "$root/online_replay.py"
  git -C "$root" init -q
  git -C "$root" add requirements.txt online_replay.py
  git -C "$root" -c user.name=test -c user.email=test@example.invalid \
    commit -qm init
}

DATA_DIR="$TMP_ROOT/shared/data"
mkdir -p "$DATA_DIR"
printf '%s\n' '{"id":"one"}' > "$DATA_DIR/input.jsonl"
printf '%s\n' '{"id":"outside"}' > "$TMP_ROOT/outside.jsonl"

GOOD_REPO="$TMP_ROOT/good"
make_repo "$GOOD_REPO"
printf '%s\n' dirty >> "$GOOD_REPO/online_replay.py"
BEFORE_HEAD="$(git -C "$GOOD_REPO" rev-parse HEAD)"
BEFORE_BRANCH="$(git -C "$GOOD_REPO" branch --show-current)"

OUTPUT="$(
  LLM_BENCH_SHARED_MOUNT="$TMP_ROOT/shared" \
  LLM_BENCH_DATASET_SRC="$DATA_DIR/input.jsonl" \
  bash "$GOOD_REPO/skill/model-perf-binary-search/scripts/bootstrap.sh" \
    --check-only
)"
eval "$OUTPUT"

test "$REPO_ROOT" = "$GOOD_REPO"
test "$DATASET" = "$DATA_DIR/input.jsonl"
test "$(git -C "$GOOD_REPO" rev-parse HEAD)" = "$BEFORE_HEAD"
test "$(git -C "$GOOD_REPO" branch --show-current)" = "$BEFORE_BRANCH"
grep -q dirty "$GOOD_REPO/online_replay.py"

BAD_REPO="$TMP_ROOT/bad"
make_repo "$BAD_REPO"
sed -i '/preselected-route/d' "$BAD_REPO/online_replay.py"
if LLM_BENCH_SHARED_MOUNT="$TMP_ROOT/shared" \
  LLM_BENCH_DATASET_SRC="$DATA_DIR/input.jsonl" \
  bash "$BAD_REPO/skill/model-perf-binary-search/scripts/bootstrap.sh" \
    --check-only; then
  echo "expected missing-flag validation to fail" >&2
  exit 1
fi

if LLM_BENCH_SHARED_MOUNT="$TMP_ROOT/shared" \
  LLM_BENCH_DATASET_SRC="$TMP_ROOT/outside.jsonl" \
  bash "$GOOD_REPO/skill/model-perf-binary-search/scripts/bootstrap.sh" \
    --check-only; then
  echo "expected outside-dataset validation to fail" >&2
  exit 1
fi

echo "[OK] repository bootstrap check-only mode"
```

- [x] Mark it executable and run it:

```bash
chmod +x skill/model-perf-binary-search/scripts/test_repo_mode.sh
bash skill/model-perf-binary-search/scripts/test_repo_mode.sh
```

Expected before implementation: FAIL because the repository copy has no
`bootstrap.sh` with `--check-only` support.

## Task 3: Implement Repository-Native Bootstrap

**Files:**

- Create `skill/model-perf-binary-search/scripts/bootstrap.sh`
- Modify `skill/model-perf-binary-search/scripts/prepare_dataset.sh`
- Modify `skill/model-perf-binary-search/scripts/test_prepare_dataset.sh`

**Steps:**

- [x] Implement these exact bootstrap interfaces:

```text
bootstrap.sh [--check-only]

Inputs:
  LLM_BENCH_DATASET_SRC       required absolute regular non-empty file
  LLM_BENCH_SHARED_MOUNT      default /mnt/shared/sss

Outputs on stdout:
  REPO_ROOT=<shell-escaped absolute path>
  DATASET=<shell-escaped canonical dataset path>
  PYTHON=<shell-escaped absolute venv Python path>
```

- [x] Resolve the root from the script, not the caller's current directory:

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
```

- [x] Validate `.git`, `requirements.txt`, `online_replay.py`, and the
  three required CLI flags. Use this failure form for each missing flag:

```bash
fail "当前 checkout 的 online_replay.py 缺少 --$flag；请使用包含回放因果功能的版本。"
```

- [x] Replace dataset copying with canonical-path validation. The dataset must
  resolve beneath `$LLM_BENCH_SHARED_MOUNT/data`; print the resolved path and
  do not create anything in `datasets/`.

- [x] Update `test_prepare_dataset.sh` to invoke:

```bash
bash "$HERE/prepare_dataset.sh" "$SHARED_DATA_DIR" "$SELECTED_FILE"
```

Expected success output:

```text
DATASET=<shell-escaped canonical selected path>
```

Retain failure cases for missing selection, empty input, an outside path, a
directory, and a symlink that resolves outside the shared data directory. Add
an assertion that no dataset file is copied beneath the temporary repository.

- [x] In `--check-only` mode, stop after validation and print assignments.
  In normal mode:

```bash
cd "$REPO_ROOT"
if ! command -v uv >/dev/null 2>&1; then
  fail "缺少 uv；请先安装 uv 后重试。"
fi
if [ ! -d .venv ]; then
  uv venv
fi
uv pip install -r requirements.txt
uv pip install requests pytest
mkdir -p bench-runs
"$REPO_ROOT/.venv/bin/python" \
  "$SCRIPT_DIR/health_check.py" --workdir "$REPO_ROOT"
```

Do not install `uv` automatically: setup may install project dependencies,
but it must not modify shell startup files or download an installer without
separate user approval.

- [x] Add a static safety assertion to the test:

```bash
if rg -n 'git (clone|fetch|checkout|pull)' "$SCRIPT_DIR/bootstrap.sh"; then
  echo "bootstrap must not mutate Git state" >&2
  exit 1
fi
```

- [x] Run the repository-mode test:

```bash
bash skill/model-perf-binary-search/scripts/test_repo_mode.sh
```

Expected: `[OK] repository bootstrap check-only mode`.

## Task 4: Write the Repository-Native Skill and References

**Files:**

- Create `skill/model-perf-binary-search/SKILL.md`
- Create
  `skill/model-perf-binary-search/references/service-lifecycle.md`
- Create `skill/model-perf-binary-search/references/tuning.md`

**Steps:**

- [x] Write this discriminating frontmatter:

```yaml
---
name: model-perf-binary-search
description: >-
  Find the maximum sustainable QPS of an OpenAI-compatible LLM service under a
  p50 end-to-end latency SLO from an already-cloned
  llm-inference-benchmarking repository. Use for SLO binary searches and
  optional serving-parameter tuning; do not use to clone or switch branches.
---
```

- [x] Make the session-start sequence explicit:

```text
1. Locate this skill inside the checked-out repository.
2. List regular files under /mnt/shared/sss/data.
3. Ask for dataset, service command, LOW/HIGH, offload ON/OFF, model, port,
   optional tuning mode, SLO, and precision.
4. Run scripts/bootstrap.sh using a path relative to this SKILL.md.
5. Read .health_check.json and apply the documented offload gate.
6. Start one service, probe LOW then HIGH, expand bounds when necessary, bisect,
   analyze every completed probe, and report the final maximum passing QPS.
```

- [x] Preserve the standard dataset modes, 8/4 versus 16/8 round policy,
  continuous conversation flags, client-E2E SLO decision, prefix-cache
  snapshots, auto-steady analyzer, runaway-stop rule, result table, and
  leave-service-running policy from the source skill.

- [x] Replace every global helper reference with a skill-relative resolution:

```bash
SKILL_DIR="$REPO_ROOT/skill/model-perf-binary-search"
"$SKILL_DIR/scripts/analyze_rounds.py"
```

- [x] Remove all branch-choice, clone/update, legacy `sss-test`, and AutoReply
  M12 instructions. The only compatibility rule is the bootstrap capability
  check.

- [x] Move conditional detail as follows:

```markdown
- Read [references/service-lifecycle.md](references/service-lifecycle.md)
  before starting or replacing a service, and for health-gate/progress rules.
- Read [references/tuning.md](references/tuning.md) only when the user opts
  into generic tuning or feature-enablement tuning.
```

- [x] Verify there are no stale external paths or excluded modes:

```bash
if rg -n '/root/|~/\.cursor|sss-test|AutoReply|git (clone|fetch|checkout|pull)' \
  skill/model-perf-binary-search; then
  exit 1
fi
```

Expected: exit 0 with no matches.

## Task 5: Integrate Smoke Tests and Ignore Generated Output

**Files:**

- Modify `skill/model-perf-binary-search/scripts/smoke.sh`
- Create `.gitignore`

**Steps:**

- [x] Append the repository-mode test to `smoke.sh` and include it in the
  final pass/fail count:

```bash
if bash "$SCRIPT_DIR/test_repo_mode.sh"; then
  pass "repository-native bootstrap"
else
  fail "repository-native bootstrap"
fi
```

- [x] Add only these ignore rules:

```gitignore
.venv/
.health_check.json
bench-runs/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
```

- [x] Run the complete copied smoke suite:

```bash
bash skill/model-perf-binary-search/scripts/smoke.sh
```

Expected: twelve checks pass and zero fail.

## Task 6: Validate the Skill and Core Replay Contract

**Files:**

- Verify all files above; no new files.

**Steps:**

- [x] Validate shell syntax:

```bash
bash -n skill/model-perf-binary-search/scripts/bootstrap.sh
bash -n skill/model-perf-binary-search/scripts/prepare_dataset.sh
bash -n skill/model-perf-binary-search/scripts/test_prepare_dataset.sh
bash -n skill/model-perf-binary-search/scripts/smoke.sh
bash -n skill/model-perf-binary-search/scripts/test_repo_mode.sh
```

Expected: all commands exit 0 with no output.

- [x] Validate Python helper syntax:

```bash
python3 -m py_compile \
  skill/model-perf-binary-search/scripts/analyze_rounds.py \
  skill/model-perf-binary-search/scripts/health_check.py \
  skill/model-perf-binary-search/scripts/prefix_cache_hit_rate.py
```

Expected: exit 0.

- [x] Run the skill package validator:

```bash
python3 /root/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  skill/model-perf-binary-search
```

Expected: validation success.

- [x] Run the retained replay tests:

```bash
.venv/bin/python -m pytest tests/test_online_replay.py -q
```

Expected: `19 passed`.

- [x] Run final scope and integrity checks:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; status shows the new skill, design/plan,
`.gitignore`, the previously approved staged deletions, and the pre-existing
`online_replay.py` modification. Do not stage, commit, push, or alter the
global Cursor skill.
