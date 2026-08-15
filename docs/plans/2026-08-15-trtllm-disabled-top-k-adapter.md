# TensorRT-LLM Disabled Top-K Adapter Implementation Plan

> **For agents:** Use `executing-plans` inline. Steps use `- [ ]` checkboxes.

**Goal:** Make the fixed top-k-disabled AutoReply workload compatible with TensorRT-LLM by omitting its `top_k` field on the wire, then complete TensorRT-LLM tuning and the fair framework final.

**Architecture:** Add an opt-in serialization policy at the existing `online_replay.py` static-extra-body seam. Add a TensorRT-specific replay builder that selects the policy while reusing all fixed workload arguments from the vLLM builder. Keep the default replay and vLLM paths unchanged, and update the benchmark skill only after a live TensorRT gate proves the route.

**Reference implementations:** `online_replay._build_extra_body`, `scripts.autoreply_vllm_tuning.build_replay_command`, `scripts.autoreply_trtllm_tuning.build_server_command`, and the existing AutoReply candidate controllers.

---

## Task 1: Lock serialization behavior with failing tests

**Files:**

- Modify `tests/test_online_replay.py`
- Modify `tests/test_autoreply_trtllm_tuning.py`

**Steps:**

- [x] Add a test showing `_build_extra_body({"extra_body": {"top_k": None, "n": 3}}, omit_none_static_extra=False)` returns both keys, including `top_k: None`.
- [x] Add a test showing the same call with `omit_none_static_extra=True` returns only `{"n": 3}`.
- [x] Add a controller test requiring the TensorRT replay command to include `--omit-none-extra-body`, encode `top_k: null` canonically, and preserve `n=3`, stop, max tokens, temperature, top-p, both penalties, and disabled min-p.
- [x] Run `.venv/bin/pytest -q tests/test_online_replay.py tests/test_autoreply_trtllm_tuning.py`; expect the new tests to fail because the flag and TensorRT builder do not exist.

## Task 2: Implement the opt-in replay adapter

**Files:**

- Modify `online_replay.py`
- Modify `scripts/autoreply_trtllm_tuning.py`

**Steps:**

- [x] Change the helper signature to `def _build_extra_body(body, *, omit_none_static_extra=False)` and filter only the static `body["extra_body"]` mapping when the option is true. Do not filter dataset sampling keys or mutate the input mapping.
- [x] Add `--omit-none-extra-body` as a default-false parser flag and propagate it through the endpoint configuration to `_build_extra_body`.
- [x] Replace the TensorRT alias of the vLLM builder with a wrapper that calls the vLLM builder, changes only the `--extra-body-json` payload from `top_k=-1` to `top_k=None`, and appends `--omit-none-extra-body`.
- [x] Run `.venv/bin/pytest -q tests/test_online_replay.py tests/test_autoreply_trtllm_tuning.py`; expect all tests to pass.

## Task 3: Prove wire serialization without a GPU

**Files:**

- Modify `tests/test_online_replay.py`

**Steps:**

- [x] Use an `httpx.MockTransport` with the installed `AsyncOpenAI` client to capture the final HTTP JSON for the TensorRT policy.
- [x] Assert the captured body has no `top_k` key, still has `n=3` and the fixed stop list, and contains no unrelated mutation.
- [x] Assert the default path with explicit `None` still emits JSON `null`, proving the new behavior is opt-in.
- [x] Run `.venv/bin/pytest -q tests/test_online_replay.py`; expect all tests to pass.

## Task 4: Run the stable-image compatibility gate

**Files:**

- Modify `bench-runs/autoreply-framework-shootout-20260814/trtllm/manifest/compatibility-summary.json`
- Create compatibility artifacts under `bench-runs/autoreply-framework-shootout-20260814/trtllm/compatibility-top-k-none/`

**Steps:**

- [x] Confirm GPU memory use is below 2,048 MiB and no task container is running; do not touch `minio_test`.
- [x] Pull `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` by its recorded digest and launch the baseline with `--max_seq_len 8192`.
- [x] Send three forms in order: `top_k=-1` (expect original 400), literal JSON null (expect schema rejection), and omitted field (expect HTTP 200).
- [x] Send non-streaming and streaming `n=3` requests through the real replay serialization path and verify indices `0,1,2`.
- [x] Record server/version/model/tokenizer/sampling evidence. Stop and clean the task container after the gate.

## Task 5: Screen TensorRT-LLM launch parameters

**Files:**

- Modify `scripts/autoreply_trtllm_tuning.py`
- Modify `configs/autoreply-trtllm-tuning-v121.json`
- Create normalized candidate results under `bench-runs/autoreply-framework-shootout-20260814/trtllm/candidates/`

**Steps:**

- [x] Re-audit the stable image CLI and extra LLM API options; keep backend and max sequence length fixed.
- [x] Run one cold baseline and bounded candidates covering batch capacity, token capacity, KV fraction, scheduler, CUDA graph, chunked prefill, and frontend/postprocess workers.
- [x] Preserve startup errors and OOMs as candidate results rather than silently dropping them.
- [x] Rank only candidates passing p50 E2E `<2.0s` at the shared screening QPS, while checking request success and `n=3`.

## Task 6: Measure formal TensorRT QPS gains

**Files:**

- Modify `scripts/autoreply_trtllm_tuning.py`
- Create formal search summaries under `bench-runs/autoreply-framework-shootout-20260814/trtllm/formal/`

**Steps:**

- [x] Run 12-round cold-start binary searches with six-round tail evaluation and 0.1-QPS precision for baseline and every finalist.
- [x] Confirm the winner three times at its best passing QPS and once at the adjacent `+0.1` failing QPS.
- [x] For every beneficial candidate, report absolute gain and percentage versus TensorRT baseline; report 0 QPS when latency improves without moving the boundary.

## Task 7: Re-run the framework final

**Files:**

- Modify `bench-runs/autoreply-framework-shootout-20260814/summary.json`
- Modify `docs/autoreply-framework-shootout-2026-08-14.md`

**Steps:**

- [x] Compare independently confirmed vLLM 9.4/9.5 against the confirmed TensorRT boundary under the semantically equivalent disabled-top-k mapping.
- [x] If boundaries tie, rank by median confirmation p50, then tail latency, error rate, variance, and configuration complexity.
- [x] Report TensorRT gain versus vLLM 9.4 in absolute QPS and percent; do not confuse internal sequence rate with external HTTP QPS.

## Task 8: Update skill only after live proof

**Files:**

- Modify `/root/.cursor/cursor-skills/skills/model-perf-binary-search/SKILL.md`

**Steps:**

- [x] Add an opt-in TensorRT-only compatibility note: canonical disabled `top_k=-1` may map to Python `None` only when None-valued extra-body keys are omitted on the wire and the server default is verified as disabled zero.
- [x] State that literal JSON null is not the same operation and that vLLM/standard flows remain unchanged.
- [x] Verify the skill commit deletes no standard-flow logic with `git diff --numstat` and an explicit deleted-line review.

## Task 9: Final verification, cleanup, commit, and push

**Files:** all files above

**Steps:**

- [x] Remove task-owned containers, duplicate raw help, temporary configs, and task-specific bytecode; keep compact manifests and formal result evidence.
- [x] Run all AutoReply tests, relevant `test_online_replay.py`, Python compilation, JSON parsing, dataset hash/row checks, confirmation assertions, container checks, and `git diff --check`.
- [x] Test the staged benchmark patch in a detached clean worktree; repeat against the exact commit after committing.
- [ ] Commit with `Saddss <2872669061@qq.com>` without changing global Git configuration.
- [ ] Push benchmark and skill commits directly to their latest remote `main` only with authenticated Saddss credentials; never force-push.
