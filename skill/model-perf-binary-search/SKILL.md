---
name: model-perf-binary-search
description: >-
  Find the maximum sustainable QPS of an OpenAI-compatible LLM service under a
  p50 end-to-end latency SLO from an already-cloned
  llm-inference-benchmarking repository. Use for SLO binary searches and
  optional serving-parameter tuning; do not use to clone or switch branches.
---

# Model Performance Binary Search

Use the checked-out repository's `online_replay.py` to find the highest HTTP
QPS whose client end-to-end p50 latency is strictly below an SLO. The bundled
bootstrap validates replay capabilities and never changes repository state.

## Session inputs

Collect all of these before setup:

1. Dataset: list regular files under `/mnt/shared/sss/data` and require the
   user to choose exactly one. Never reuse or infer a prior selection.
2. Full service startup command, including the listening port.
3. Initial QPS bounds as `LOW HIGH`.
4. Offload `ON` or `OFF`; ask explicitly instead of inferring it.
5. Model name passed to `--model`.
6. API base port.
7. Whether to run tuning after the baseline: none, generic parameter tuning,
   or feature-enablement tuning.
8. Optional SLO and precision overrides. Defaults are `6.5` seconds and
   `0.1` QPS.

Do not start a live benchmark while any required input is missing.

## Repository setup

Resolve `SKILL_DIR` to the directory containing this file. Then run:

```bash
export LLM_BENCH_DATASET_SRC="/mnt/shared/sss/data/chosen-file"
eval "$(bash "$SKILL_DIR/scripts/bootstrap.sh")"
```

The command returns shell-safe `REPO_ROOT`, `DATASET`, and `PYTHON`.
Always run later repository commands from `$REPO_ROOT`; invoke helpers below
`$SKILL_DIR/scripts`.

Bootstrap requires the checked-out `online_replay.py` to expose
`--serialize-conversations`, `--continuous-qps-window`, and
`--preselected-route`. A failed capability check is a hard stop.

Read `$REPO_ROOT/.health_check.json` immediately:

- `exit=0`: proceed.
- `exit=1`: paste `issues_warn` verbatim. Non-offload may proceed; obtain
  confirmation before offload.
- `exit=2`: paste `issues_red` and `issues_warn` verbatim. Refuse offload
  unless the user explicitly supplies `force=true`; non-offload may proceed
  with the warning recorded.

Before starting or replacing any service, read
[references/service-lifecycle.md](references/service-lifecycle.md).
If tuning was selected, also read
[references/tuning.md](references/tuning.md).

## Dataset execution mode

Use exactly one routing mode:

- `kaon-v3-test.jsonl`: for target QPS `q`, pass
  `--sample-range 0.0 min(0.02*q, 1.0)`.
- `gemma4-31b-test.jsonl`: pass `--preselected-route` and never
  `--sample-range`.
- Any other dataset: inspect provenance and explicitly decide whether it is a
  hash-sampled population or a complete preselected route. Stop if unknown.

Always pass `--serialize-conversations --continuous-qps-window`. Same-
conversation requests remain ordered while different conversations overlap,
and sequencing wait remains part of client E2E latency.

For truncation-aware data, forward `body.enable_kv_evict` only when the user
explicitly requests eviction testing by adding `--forward-kv-evict`. For MTP
runs add `--disable-min-p` and omit `--min-p`.

## Fixed probe policy

- Replay mode: `qps`.
- Round duration: 30 seconds.
- Offload OFF: 8 rounds, legacy tail window 4, progress every 15 minutes.
- Offload ON: 16 rounds, legacy tail window 8, progress every 30 minutes.
- Production defaults when the dataset does not carry request fields:
  `--max-tokens 200 --temperature 0.7 --top-p 0.85 --top-k 40
  --min-p 0.1 --frequency-penalty 0.4 --presence-penalty 0.1`.
- Use `--json-output` for per-round metrics.
- Decide PASS/FAIL from client E2E latency, never server latency alone.
- Run every required round unless the runaway safety rule below applies.

Create unique files beneath `$REPO_ROOT/bench-runs` for each probe:

```text
qps_<q>_<timestamp>_shard<n>.jsonl
qps_<q>_<timestamp>.before.prom
qps_<q>_<timestamp>.after.prom
```

## Prefix-cache measurement

Before traffic, snapshot `http://localhost:<port>/metrics`:

```bash
"$PYTHON" "$SKILL_DIR/scripts/prefix_cache_hit_rate.py" snapshot --url "http://localhost:$PORT/metrics" --out "$BEFORE_PROM"
```

After all shards exit, take the second snapshot and diff them:

```bash
"$PYTHON" "$SKILL_DIR/scripts/prefix_cache_hit_rate.py" snapshot --url "http://localhost:$PORT/metrics" --out "$AFTER_PROM"
"$PYTHON" "$SKILL_DIR/scripts/prefix_cache_hit_rate.py" diff --before "$BEFORE_PROM" --after "$AFTER_PROM"
```

If no prefix counters are exposed, try service-log lines containing a prefix
cache hit rate, then documented engine-specific metrics. If neither works,
record `n/a` and continue; cache observability does not block the capacity
search.

## Running one probe

Use one replay process for `q <= 10`. A hash-sampled population may use
`ceil(q/10)` parallel shards above 10 QPS, each with target `q/n` and a
non-overlapping sample-range chunk. Never range-shard a preselected route.

The base command is:

```bash
cd "$REPO_ROOT"
"$PYTHON" online_replay.py --input "$DATASET" --preload-time 2 --replay-mode qps --target-qps "$QPS" --serialize-conversations --continuous-qps-window --api-base "http://localhost:$PORT/v1" --api-key aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa --model "$MODEL" --use-chat --max-tokens 200 --temperature 0.7 --top-p 0.85 --top-k 40 --min-p 0.1 --frequency-penalty 0.4 --presence-penalty 0.1 --round-duration 30 --round-drain-timeout 300 --request-timeout 600 --max-rounds "$TOTAL_ROUNDS" --e2e-slo "$SLO" --json-output "$ROUND_JSON"
```

Add exactly one dataset-mode argument. Wait for every shard to exit, then
analyze:

```bash
"$PYTHON" "$SKILL_DIR/scripts/analyze_rounds.py" --json "$ROUND_JSON" --total-rounds "$TOTAL_ROUNDS" --tail-window "$TAIL_WINDOW" --slo "$SLO" --auto-steady
```

The analyzer exits `0=PASS`, `1=FAIL`, and
`2=NOT_ENOUGH_ROUNDS`. Count `NOT_ENOUGH_ROUNDS` as FAIL and preserve its
JSON for diagnosis. The auto-detected steady average is the primary signal
when it contains at least three rounds; otherwise the tail average is primary.
Always report both metrics and any `warmup_dominated` note.

## Search algorithm

Round every candidate to the selected precision grid and never infer an
untested boundary.

1. Probe LOW first.
2. If LOW fails, move downward: halve it when p50 is over `1.5*SLO`;
   otherwise subtract `max(0.5, 0.3*LOW)`. Stop if the precision floor fails.
3. Probe HIGH after establishing a passing LOW.
4. If HIGH passes, raise it according to measured SLO slack:
   - slack at least 40%: `min(1.6*HIGH, HIGH+8)`
   - slack 20%-40%: `1.3*HIGH`
   - slack 5%-20%: `HIGH + max(1.0, 0.15*HIGH)`
   - slack below 5%: probe a nearby higher point; do not declare it failed
     without running it.
5. Once one bound passes and the other fails, probe the rounded midpoint until
   the bracket width is at most the requested precision.
6. The answer is the largest actually tested QPS whose primary p50 is strictly
   less than the SLO.

## Runaway safety stop

Early-stop only when:

- per-round p50 rises monotonically, the latest p50 exceeds `10*SLO`, and
  the engine is clearly saturated; or
- at least two consecutive drain-timeout rounds have zero successes, or at
  least 50 ReadTimeouts occur before three metric rounds exist.

Mark the probe FAIL, record why it was stopped, confirm the service is actually
alive, and continue below that QPS. A service OOM under load is a QPS failure
when a lower point already passed; relaunch only with user authorization.

## Reporting

Track every probe:

| Step | QPS | Result | Primary p50 (window) | Tail-N p50 | Prefix cache hit | Notes |
|------|-----|--------|----------------------|------------|------------------|-------|

Finish with:

```text
Max QPS meeting p50 e2e <SLO>s: <best_pass>
Offload: ON|OFF; rounds: 8|16; tail: 4|8
```

List all round JSON paths, the service PID or container name, and its log path.
State that the service remains running. Do not commit benchmark outputs.
