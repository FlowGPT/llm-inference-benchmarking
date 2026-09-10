# Replay KV-Eviction Forwarding Implementation Plan

> **For agents:** Use `executing-plans` inline with test-first changes.

**Goal:** Add opt-in forwarding of per-request truncation flags from replay
datasets to the vLLM OpenAI API.

**Architecture:** Keep truncation classification in JSONL
`body.enable_kv_evict`. Restore the tracked replay client and gate copying that
field into OpenAI `extra_body` behind one CLI boolean.

**Reference implementation:** vLLM
`ChatCompletionRequest.to_sampling_params()` on `feat/kv-evict-trunc`.

---

## Task 1: Restore the tracked replay client

**Files:**
- Restore: `online_replay.py`

**Steps:**
- [ ] Restore `online_replay.py` from `HEAD`.
- [ ] Confirm the branch sampling, timeout, and QPS-ordering changes remain.

## Task 2: Add forwarding behavior test-first

**Files:**
- Modify: `online_replay.py`
- Modify or create: `tests/test_online_replay.py`

**Steps:**
- [ ] Add failing tests proving default omission, enabled true/false
  forwarding, conversation header preservation, and no nested truncation KV
  parameters.
- [ ] Run `.venv/bin/python -m pytest tests/test_online_replay.py -q` and
  confirm failures.
- [ ] Add `--forward-kv-evict`, default false.
- [ ] Copy `body.enable_kv_evict` into request `extra_body` only when enabled.
- [ ] Add `--disable-min-p` without changing its default probe behavior.
- [ ] Re-run the focused tests and confirm all pass.

## Task 3: Validate the aligned dataset

**Files:**
- Validate:
  `data/online-logs-20260716/gemma31_chatservice_replay_identity_aligned_6247_320.jsonl`

**Steps:**
- [ ] Assert all 320 requests contain a boolean `body.enable_kv_evict`.
- [ ] Confirm request count and truncation rate remain unchanged.

## Task 4: Review and live verification

**Files:**
- Review the complete working-tree diff.

**Steps:**
- [ ] Run the review skill against `HEAD`, checking standards and this spec.
- [ ] Fix blocking findings.
- [ ] Run focused tests and lint checks.
- [ ] Run a cold live probe with forwarding disabled and enabled.
- [ ] Verify enabled mode reaches vLLM's standard header-plus-flag path.
- [ ] Commit on `sss-test`, verify no forbidden attribution, and push only
  after the requested author identity is confirmed.
