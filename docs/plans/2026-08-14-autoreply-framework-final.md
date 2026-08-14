# AutoReply Cross-Framework Final Implementation Plan

> **For agents:** Use `executing-plans` inline. Steps use `- [ ]` checkboxes.

**Goal:** Reproduce and fairly compare the confirmed vLLM and TensorRT-LLM
champions, then publish one winning framework configuration.

**Architecture:** Add a read-mostly final controller that accepts only
authoritative champion manifests, rebuilds their commands, alternates cold
runs, validates invariant fingerprints, and ranks normalized results. It does
not tune parameters or modify champion configurations.

**Reference implementations:** The two framework tuning controllers and their
normalized candidate-result schema.

---

## File structure

- Create `scripts/autoreply_framework_final.py`: manifest validation,
  interleaved execution, result ranking, and report generation.
- Create `tests/test_autoreply_framework_final.py`: invariant, ordering,
  eligibility, and tie-break tests.
- Create `docs/autoreply-framework-shootout-2026-08-14.md`: final report.

## Task 1: Validate champion manifests

**Files:** Create both script and test file above.

**Steps:**

- [ ] Write failing tests that reject different dataset/model/request hashes,
  missing cold confirmation, missing adjacent FAIL, non-FP8 KV, or non-0.1 QPS
  boundaries.
- [ ] Implement manifest parsing and invariant fingerprint comparison.
- [ ] Run `.venv/bin/pytest -q tests/test_autoreply_framework_final.py`;
  expect PASS after implementation.

## Task 2: Implement interleaved cold execution

**Files:** Modify both files above.

**Steps:**

- [ ] Test the exact order `vllm,trtllm,trtllm,vllm,vllm,trtllm`, fresh-server
  enforcement, framework-owned cleanup, and three runs per framework.
- [ ] Rebuild launch commands from redacted structured arguments rather than
  shell strings and run the unchanged replay builder at each claimed max QPS.
- [ ] Persist complete per-run evidence and normalized cache metrics.
- [ ] Run unit tests; expect PASS.

## Task 3: Rank and report

**Files:** Modify the controller and create the final report.

**Steps:**

- [ ] Test that all three runs must pass and tie-break order is QPS, median
  primary p50, tail p50, error rate, variance, then simplicity.
- [ ] Execute the six-run final and recheck each champion's max+0.1 FAIL if its
  earlier observation is not reusable.
- [ ] Generate the final report with exact winning command, absolute and
  percentage gain over 9.3 QPS, cache normalization, all raw artifact paths,
  and incompatibility caveats.
- [ ] Run the final controller tests, full relevant pytest selection, dataset
  verifier, and `git diff --check`; expect PASS.
