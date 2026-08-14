# AutoReply vLLM Max-QPS Tuning Implementation Plan

> **For agents:** Use `executing-plans` inline. Steps use `- [ ]` checkboxes.

**Goal:** Build and run a resumable vLLM 0.27.1 parameter-family search that
finds the highest one-GPU AutoReply HTTP QPS under a strict 2-second primary
p50 E2E SLO without changing the model or request sampling parameters.

**Architecture:** Add one AutoReply-specific Python controller modeled after
`run_kaon_v3_eight_config_matrix.py`, but keep request generation self-contained
so its fixed `n=3` sampling contract cannot inherit generic benchmark defaults.
The controller captures the target CLI, inventories every flag, launches only
task-owned containers, runs cold screening and full binary searches, and writes
atomic resumable state plus normalized evidence under one artifact root.

**Reference implementations:** `run_kaon_v3_eight_config_matrix.py` for Docker
lifecycle, state recovery, and binary search; `run_gemma4_mtp_matrix.py` for
metrics snapshots and probe monitoring; `scripts/run_autoreply_probe.sh` for
the exact AutoReply replay request.

---

## File structure

- Create `scripts/autoreply_vllm_tuning.py`: candidate definitions, CLI
  inventory, safe Docker lifecycle, exact replay probe, screening, finalist
  binary search, resume state, and report generation.
- Create `tests/test_autoreply_vllm_tuning.py`: unit tests for invariants,
  candidate commands, classification, selection, resume, and binary search.
- Create `configs/autoreply-vllm-tuning-v0271.json`: explicit ordered candidate
  families and values; no API key or other secret.
- Modify `scripts/run_autoreply_probe.sh`: accept optional artifact/round/QPS
  settings through named environment variables while leaving all sampling
  arguments literal and unchanged.
- Create `docs/autoreply-vllm-tuning-2026-08-14.md`: generated final report
  written only after the run completes.

## Task 1: Lock request and model invariants in tests

**Files:**

- Create `tests/test_autoreply_vllm_tuning.py`.
- Create `scripts/autoreply_vllm_tuning.py` with constants and pure builders.

**Steps:**

- [ ] Add a failing test that imports `build_replay_command()` and asserts the
  exact immutable request fragment:

```python
def test_replay_command_preserves_autoreply_contract(tmp_path):
    command = tuning.build_replay_command(
        qps=9.4, rounds=6, output=tmp_path / "probe.jsonl"
    )
    joined = " ".join(command)
    assert "--max-tokens 50" in joined
    assert "--temperature 0.7" in joined
    assert "--top-p 0.8" in joined
    assert "--frequency-penalty 0.01" in joined
    assert "--presence-penalty 0.01" in joined
    assert "--disable-min-p" in command
    assert json.loads(command[command.index("--extra-body-json") + 1]) == {
        "n": 3,
        "stop": ["<|im_end|>"],
        "top_k": -1,
    }
    assert "--preselected-route" in command
    assert "--sample-range" not in command
```

- [ ] Add a failing test that `build_server_command()` always retains the exact
  model path, served name, `--quantization modelopt`, and max model length 8192.
- [ ] Run:

```bash
cd /root/llm-inference-benchmarking
.venv/bin/pytest -q tests/test_autoreply_vllm_tuning.py
```

Expected: import failure because the controller does not exist.

- [ ] Implement immutable constants, `Candidate`, `build_replay_command()`, and
  `build_server_command()` minimally, using list arguments rather than shell
  interpolation.
- [ ] Re-run the test; expected PASS.

## Task 2: Inventory and classify every target CLI flag

**Files:**

- Modify `scripts/autoreply_vllm_tuning.py`.
- Modify `tests/test_autoreply_vllm_tuning.py`.

**Steps:**

- [ ] Add tests for `extract_flags(help_text)` and `classify_flag(flag)`:

```python
def test_flag_inventory_has_explicit_disposition():
    flags = tuning.extract_flags(
        "usage: api_server.py [--max-num-seqs N] [--enable-lora] [--api-key K]"
    )
    rows = [tuning.classify_flag(flag) for flag in flags]
    assert {row["flag"] for row in rows} == {
        "--api-key", "--enable-lora", "--max-num-seqs"
    }
    assert all(row["class"] and row["reason"] for row in rows)
```

- [ ] Add classification rules for fixed invariants, applicable search/control,
  not-applicable families, and unsafe/unsupported flags. Unknown flags must be
  classified `review-required`, never silently ignored.
- [ ] Add `capture_manifest()` that runs the target image with `--gpus all` and
  `--help=all`, writes raw help/version/image digest, and atomically writes
  `flag_compatibility.json` plus `flag_compatibility.md`.
- [ ] Test with fixture help text; expected PASS without Docker/GPU.

## Task 3: Define bounded parameter families and safe server commands

**Files:**

- Create `configs/autoreply-vllm-tuning-v0271.json`.
- Modify `scripts/autoreply_vllm_tuning.py`.
- Modify `tests/test_autoreply_vllm_tuning.py`.

**Steps:**

- [ ] Define ordered families for sequence/batched/scheduled token capacity,
  microbatching, prefill scheduling, GPU memory, KV layout/dtype, prefix
  matching/sharing, kernels, compilation/CUDA graphs, allocator, API frontend,
  streaming, and target-only speculative decoding.
- [ ] Represent every candidate as a name, parent, family, hypothesis, argument
  diff, compatibility predicates, and risk label. Example:

```json
{
  "name": "seq128",
  "parent": "baseline",
  "family": "sequence_capacity",
  "hypothesis": "n=3 needs more child-sequence admission headroom",
  "args": ["--max-num-seqs", "128"],
  "risk": "queueing_or_graph_pressure"
}
```

- [ ] Add tests that candidate names are unique, every applicable family has at
  least one candidate or an exclusion reason, no candidate changes fixed model
  flags, only task-owned container names are generated, and secrets are redacted
  from persisted commands.
- [ ] Implement compatibility filtering against the captured flag inventory.
  Unsupported candidates become durable `SKIPPED` results with reasons.
- [ ] Run tests; expected PASS.

## Task 4: Implement one cold screening probe end to end

**Files:**

- Modify `scripts/autoreply_vllm_tuning.py`.
- Modify `scripts/run_autoreply_probe.sh`.
- Modify `tests/test_autoreply_vllm_tuning.py`.

**Steps:**

- [ ] Add mocked lifecycle tests proving the controller stops/removes only an
  exact `autoreply-m12-tune-*` container and pauses when unrelated GPU use is
  above 2,048 MiB.
- [ ] Parameterize only output directory, rounds, tail window, API base/key
  source, and QPS in `run_autoreply_probe.sh`; keep sampling arguments literal.
- [ ] Implement atomic candidate artifacts: redacted command, container id,
  service log, readiness, resolved `/metrics` cache config, before/after
  Prometheus snapshots, client JSONL/log/return code, prefix diff, analyzer
  JSON, and normalized result.
- [ ] Enforce gates: readiness, three non-streaming choices, streaming indices
  0/1/2, 100% request success for selection, primary p50 `<2.0`, and prefix hit
  in `[0.66, 0.67]` for cold screening.
- [ ] Run unit tests, then run only the baseline six-round 9.4-QPS screen.
  Expected: complete normalized result even if SLO status is FAIL.

## Task 5: Add resumable Stage-1 and Stage-2 search

**Files:**

- Modify `scripts/autoreply_vllm_tuning.py`.
- Modify `tests/test_autoreply_vllm_tuning.py`.

**Steps:**

- [ ] Add tests for atomic `state.json`, adoption of complete results, retry of
  incomplete active candidates, durable failed/skipped candidates, and exact
  one-at-a-time candidate execution.
- [ ] Implement Stage 1 one-family screening from the JSON config. Carry a
  family winner forward only after its cold result is valid.
- [ ] Implement Stage 2 coordinate descent with the six required interactions
  from the approved spec. Retain a change only on a 2% p50 improvement, FAIL to
  PASS transition, stability improvement, or higher fixed-QPS screen.
- [ ] Rank and persist three materially distinct finalists.
- [ ] Unit-test deterministic ranking and finalist diversity; expected PASS.

## Task 6: Add finalist binary search and cold confirmation

**Files:**

- Modify `scripts/autoreply_vllm_tuning.py`.
- Modify `tests/test_autoreply_vllm_tuning.py`.

**Steps:**

- [ ] Add a fake probe map and tests proving LOW=8.5 runs before HIGH=10.5,
  upward/downward extrapolation works, 0.1 precision converges, incomplete
  rounds are FAIL, and a load-killed server is relaunched below a confirmed
  PASS.
- [ ] Implement 12-round/tail-6 full probes for three finalists, reusing the
  same exact request builder and per-probe prefix snapshots.
- [ ] Implement at most two evidence-driven adjacent refinements.
- [ ] Select the highest confirmed QPS, cold-confirm it, and ensure the adjacent
  `+0.1` point has a valid FAIL result.
- [ ] Run unit tests; expected PASS.

## Task 7: Generate the comparison report and verify the controller

**Files:**

- Modify `scripts/autoreply_vllm_tuning.py`.
- Create `docs/autoreply-vllm-tuning-2026-08-14.md` at run completion.

**Steps:**

- [ ] Generate the complete flag disposition matrix, all screening/interaction
  rows, finalist probe tables, baseline/winner delta, exact redacted winning
  command, failure/exclusion reasons, artifact links, and final container state.
- [ ] Run:

```bash
cd /root/llm-inference-benchmarking
.venv/bin/pytest -q tests/test_autoreply_vllm_tuning.py
.venv/bin/python scripts/verify_autoreply_dataset.py
git diff --check -- scripts/autoreply_vllm_tuning.py \
  scripts/run_autoreply_probe.sh tests/test_autoreply_vllm_tuning.py \
  configs/autoreply-vllm-tuning-v0271.json
```

Expected: all tests PASS, dataset report status PASS, and no whitespace errors.

## Task 8: Execute and monitor the long run

**Files:**

- Runtime artifacts under
  `bench-runs/autoreply-tuning-20260814/`.
- Final report `docs/autoreply-vllm-tuning-2026-08-14.md`.

**Steps:**

- [ ] Rebuild `datasets/autoreply_prod_dist_repeated_13x.jsonl` and verify its
  13,000 rows.
- [ ] Run the health check and copy its exact warnings into the run manifest.
- [ ] Start the resumable controller in a detached session and record its PID,
  controller log, and `state.json` path.
- [ ] Monitor at least every 15 minutes, immediately surfacing service crashes,
  unrelated GPU occupation, invalid prefix hit rate, or request-shape failure.
- [ ] Continue through screening, interactions, three finalist searches,
  refinement, and winner confirmation.
- [ ] Run fresh final verification against authoritative result files before
  claiming a winner. Leave only the winning task container running and do not
  touch `minio_test`.
