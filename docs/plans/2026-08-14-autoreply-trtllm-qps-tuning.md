# AutoReply TensorRT-LLM Max-QPS Tuning Implementation Plan

> **For agents:** Use `executing-plans` inline. Steps use `- [ ]` checkboxes.

**Goal:** Build and run a resumable TensorRT-LLM parameter search that finds
the highest one-GPU AutoReply HTTP QPS under the fixed two-second p50 SLO.

**Architecture:** Add a TensorRT-LLM controller beside the vLLM controller and
reuse its immutable replay builder, analyzer, artifact schema, GPU ownership
guard, and binary-search primitives. Keep framework-specific image discovery,
CLI classification, launch construction, readiness, and metrics normalization
behind a small adapter so the comparison consumes identical normalized results.

**Reference implementations:** `scripts/autoreply_vllm_tuning.py` after its
implementation, `run_kaon_v3_eight_config_matrix.py` for lifecycle/resume, and
`scripts/run_autoreply_probe.sh` for the fixed request contract.

---

## File structure

- Create `scripts/autoreply_trtllm_tuning.py`: stable/RC preflight, CLI
  inventory, candidate generation, safe lifecycle, screening, binary search,
  and report generation.
- Create `tests/test_autoreply_trtllm_tuning.py`: compatibility, invariants,
  config, resume, metrics, and selection tests.
- Create `configs/autoreply-trtllm-tuning-v121.json`: ordered candidate
  families and bounded values.
- Create `docs/autoreply-trtllm-tuning-2026-08-14.md`: generated run report.

## Task 1: Lock the image, backend, model, and request gates

**Files:**

- Create `scripts/autoreply_trtllm_tuning.py`.
- Create `tests/test_autoreply_trtllm_tuning.py`.

**Steps:**

- [ ] Write a failing test asserting stable image `1.2.1`, fallback
  `1.3.0rc22`, exact model path, explicit `--backend pytorch`, and no removed
  engine-build command or TensorRT backend.
- [ ] Assert the controller imports the exact replay builder used by vLLM and
  cannot override any sampling field.
- [ ] Run `.venv/bin/pytest -q tests/test_autoreply_trtllm_tuning.py`; expect
  import failure.
- [ ] Implement immutable constants and pure image/server-command builders.
- [ ] Re-run the test; expect PASS.

## Task 2: Capture the actual CLI and compatibility manifest

**Files:**

- Modify `scripts/autoreply_trtllm_tuning.py`.
- Modify `tests/test_autoreply_trtllm_tuning.py`.

**Steps:**

- [ ] Add fixture tests for parsing `trtllm-serve --help`, resolved versions,
  and classifying every flag with an explicit reason. Unknown flags resolve to
  `review-required`.
- [ ] Implement image pull/digest capture and run the target image to capture
  help, TensorRT-LLM/PyTorch/CUDA/ModelOpt versions, GPU, and driver.
- [ ] Persist raw and Markdown/JSON normalized manifests atomically.
- [ ] Run the unit tests; expect PASS without needing a live server.

## Task 3: Implement the stable-to-RC compatibility gate

**Files:**

- Modify `scripts/autoreply_trtllm_tuning.py`.
- Modify `tests/test_autoreply_trtllm_tuning.py`.

**Steps:**

- [ ] Add mocked tests proving RC is attempted only after a durable stable
  failure and is never used merely because it benchmarks faster.
- [ ] Implement exact-checkpoint launch, readiness, `/v1/models`, `/metrics`,
  three-choice non-streaming, three-index streaming, finite-output, FP8-KV,
  and memory-stability checks.
- [ ] Persist redacted requests, structural response summaries, logs, and the
  selected image digest.
- [ ] Run the live compatibility gate. Expected result is either stable PASS,
  RC fallback PASS, or a durable framework-INCOMPATIBLE report.

## Task 4: Define and validate the bounded search space

**Files:**

- Create `configs/autoreply-trtllm-tuning-v121.json`.
- Modify `scripts/autoreply_trtllm_tuning.py`.
- Modify `tests/test_autoreply_trtllm_tuning.py`.

**Steps:**

- [ ] Define candidate families for batch/sequence/token capacity, KV cache,
  scheduler/overlap/chunked prefill, CUDA Graph, kernels/sampler/compile,
  frontend workers, cache reuse, and metrics intervals.
- [ ] Give every candidate a name, parent, family, hypothesis, config diff,
  compatibility predicate, and risk.
- [ ] Test that each applicable CLI family has candidates or an exclusion,
  fixed fields cannot change, names are unique, and unsupported settings become
  durable SKIPPED records.
- [ ] Run the unit tests; expect PASS.

## Task 5: Run one cold baseline screen end to end

**Files:**

- Modify `scripts/autoreply_trtllm_tuning.py`.
- Modify `tests/test_autoreply_trtllm_tuning.py`.

**Steps:**

- [ ] Add mocked lifecycle tests proving only exact
  `autoreply-m12-trtllm-*` containers can be stopped and unrelated GPU use
  pauses the run.
- [ ] Implement the same atomic artifact set as vLLM: command, logs, readiness,
  metrics before/after, client output, analyzer output, cache normalization,
  and normalized result.
- [ ] Enforce 100% success, three choices, p50 `<2.0`, fixed sampling, and
  normalized prefix hit `[0.66, 0.67]`.
- [ ] Run a six-round 9.4-QPS baseline screen; require a complete result even
  when status is FAIL.

## Task 6: Add screening, interactions, and resumability

**Files:**

- Modify `scripts/autoreply_trtllm_tuning.py`.
- Modify `tests/test_autoreply_trtllm_tuning.py`.

**Steps:**

- [ ] Test atomic state adoption, incomplete-candidate retry, durable failure,
  one-candidate-at-a-time execution, and deterministic candidate ranking.
- [ ] Implement 24-36 one-family/interaction launches with the spec's bounded
  extension rule and coordinate descent.
- [ ] Promote three materially distinct finalists and persist the selection
  rationale.
- [ ] Run the unit tests; expect PASS.

## Task 7: Add full QPS search and winner confirmation

**Files:**

- Modify `scripts/autoreply_trtllm_tuning.py`.
- Modify `tests/test_autoreply_trtllm_tuning.py`.

**Steps:**

- [ ] Test LOW-first probing, high/low extrapolation, 0.1 precision, killed
  server recovery, and adjacent-failure enforcement using a fake probe map.
- [ ] Implement 12-round/tail-six binary search for three finalists plus at
  most two evidence-driven refinements.
- [ ] Cold-confirm the winner at max QPS and ensure max+0.1 has a valid FAIL.
- [ ] Generate `docs/autoreply-trtllm-tuning-2026-08-14.md` with every command,
  result, failure, exclusion, and artifact path.
- [ ] Run the full unit suite and dataset verifier; expect PASS.

## Task 8: Execute and monitor the long run

**Files:**

- Runtime artifacts under
  `bench-runs/autoreply-framework-shootout-20260814/trtllm/`.
- Final TensorRT-LLM report.

**Steps:**

- [ ] Start the resumable controller and record its process/session identity.
- [ ] Monitor progress and immediately preserve startup failures, OOMs,
  invalid request shapes, cache misalignment, and unrelated GPU occupation.
- [ ] Continue until compatibility failure is proven or a cold-confirmed
  TensorRT-LLM QPS boundary exists.
