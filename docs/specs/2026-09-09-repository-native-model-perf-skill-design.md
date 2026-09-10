# Repository-Native Model Performance Skill Design

## Goal

Add a self-contained copy of `model-perf-binary-search` to this repository so
an agent can clone `llm-inference-benchmarking`, invoke the bundled skill, and
run the benchmark workflow without relying on `~/.cursor/skills` or changing
Git branches.

The existing global skill at
`/root/.cursor/skills/model-perf-binary-search` remains unchanged.

## Scope

The repository skill supports the standard SLO-based maximum-QPS workflow:

1. Validate the checked-out repository and benchmark capabilities.
2. Select one explicit dataset.
3. Prepare the Python environment and run the health check.
4. Start the user-provided OpenAI-compatible service.
5. Probe low and high QPS, then binary-search to the requested precision.
6. Capture prefix-cache metrics and analyze steady-state p50 latency.
7. Report the maximum passing QPS and leave the service running.

The stale AutoReply M12 workflow is excluded because its builders, launchers,
datasets, and verification scripts are not part of the cleaned feature.

## Repository Layout

```text
skill/
└── model-perf-binary-search/
    ├── SKILL.md
    ├── references/
    │   ├── service-lifecycle.md
    │   └── tuning.md
    └── scripts/
        ├── analyze_rounds.py
        ├── bootstrap.sh
        ├── health_check.py
        ├── prefix_cache_hit_rate.py
        ├── prepare_dataset.sh
        ├── smoke.sh
        ├── test_repo_mode.sh
        └── fixtures/
```

`SKILL.md` contains the entry conditions, required user inputs, fixed
benchmark invariants, and binary-search procedure. Detailed service management
and optional tuning guidance move into references so normal invocations do not
load unrelated instructions.

## Repository Discovery and Safety

`bootstrap.sh` derives the repository root from its own location and verifies
that it contains `online_replay.py`, `requirements.txt`, and `.git`.
It never runs `git clone`, `git fetch`, `git checkout`, `git pull`, or
any command that changes repository history or the selected branch.

Compatibility is capability-based rather than branch-name-based. The bootstrap
requires `online_replay.py` to expose:

- `--serialize-conversations`
- `--continuous-qps-window`
- `--preselected-route`

This permits renamed or downstream branches while rejecting the legacy
`sss-test` implementation with an actionable error.

The script must tolerate a dirty working tree and must not stage, discard, or
rewrite user changes.

## Environment Setup

The repository root is the benchmark work directory. Bootstrap:

1. Uses an existing `.venv` or creates one with `uv`.
2. Installs `requirements.txt` plus the helper dependency `requests`.
3. Creates `bench-runs/`.
4. Runs the bundled `health_check.py` and writes `.health_check.json`.
5. Prints shell-safe `REPO_ROOT`, `DATASET`, and `PYTHON` assignments.

An explicit check-only mode validates repository discovery, required flags,
dataset selection, and output paths without installing dependencies or running
GPU checks. Tests use this mode.

## Dataset Handling

At every benchmark session the agent lists regular files under
`/mnt/shared/sss/data` and asks the user to choose exactly one. A missing,
empty, relative, or out-of-directory path fails before environment setup.

The selected shared file is used directly as `--input`; it is not copied into
the repository. This avoids duplicating multi-gigabyte datasets. Dataset mode
remains exclusive:

- `kaon-v3-test.jsonl`: hash-sampled with
  `--sample-range 0 min(0.02*qps, 1.0)`.
- `gemma4-31b-test.jsonl`: complete canonical route with
  `--preselected-route`, never `--sample-range`.
- Other datasets: inspect provenance and explicitly choose one mode.

## Benchmark Invariants

Every probe uses the checked-out `online_replay.py` through the repository
`.venv`, with:

- `--serialize-conversations --continuous-qps-window`
- QPS replay and 30-second reporting rounds
- client E2E latency as the SLO signal
- eight rounds and tail four when offload is disabled
- sixteen rounds and tail eight when offload is enabled
- `--auto-steady` analysis by default
- LOW probed before HIGH
- upward or downward bound expansion when the initial bounds do not bracket
  capacity
- complete probes except the documented runaway-queue safety stop

Service start, readiness, ownership, Docker handling, progress updates, and
optional parameter/feature tuning retain the original behavior, with every
helper path resolved relative to the bundled skill.

## Generated Files

Add narrowly scoped ignore rules for:

- `.venv/`
- `.health_check.json`
- `bench-runs/`
- Python and pytest caches

The skill never deletes existing benchmark output. Reports remain available
locally but do not enter commits accidentally.

## Validation

The copied helper smoke suite must still pass its existing eleven checks.
`test_repo_mode.sh` adds isolated coverage for:

- locating the repository from the bundled skill directory;
- succeeding on a compatible checked-out repository;
- rejecting a repository missing a required replay flag;
- rejecting invalid dataset paths;
- preserving the current branch and dirty working tree;
- avoiding all clone, fetch, checkout, and pull operations.

Run the skill validator against the final directory and run
`tests/test_online_replay.py` to confirm the benchmark entry point remains
compatible.

## Non-Goals

- Modifying the global Cursor skill.
- Selecting or switching Git branches.
- Restoring AutoReply-specific benchmark assets.
- Bundling datasets or benchmark results.
- Starting a live benchmark without the user's service command, bounds, model,
  port, offload choice, and explicit dataset selection.
