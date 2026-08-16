# AutoReply Parallel-Draft Proxy Benchmark Implementation Plan

> **For agents:** Use `executing-plans` inline. Steps use `- [ ]` checkboxes.

**Goal:** Build structurally valid EAGLE3, DFlash, and DSpark proxy draft
checkpoints, measure synthetic 50%-60% acceptance with BF16 and supported draft
quantization, and produce a numerical train/no-train recommendation under the
fixed AutoReply workload.

**Architecture:** Extend the existing benchmark repository with one deterministic
proxy-checkpoint builder and one resumable experiment controller. Reuse vLLM's
native model registry, native synthetic rejection sampler, existing
`online_replay.py`, Prometheus counters, and the established cold-launch artifact
layout. Do not patch vLLM unless a native DFlash/DSpark smoke identifies one
specific compatibility seam; patched diagnostics remain separate.

**Reference implementations:**

- `bench-runs/autoreply-eagle-smoke-20260815/draft/` for the compatible classic
  EAGLE donor tensors and config dimensions.
- `docs/plans/2026-08-15-autoreply-synthetic-eagle-acceptance.md` for fixed
  request arguments and acceptance-counter validation.
- vLLM 0.27.1 `llama_eagle3.py`, `qwen3_dflash.py`, and `qwen3_dspark.py` for
  checkpoint keys, loader behavior, and native proposer paths.
- `bench-runs/autoreply-ngram-20260816/` for 30-second screening, three-run
  confirmations, adjacent failure, and compact report structure.

---

## File structure

- Create `scripts/build_parallel_draft_proxy.py`: deterministic config and
  safetensors construction; dry-run tensor manifest; no benchmark lifecycle.
- Create `tests/test_parallel_draft_proxy.py`: config, schedule, tensor-shape,
  and manifest unit tests that run without a GPU.
- Create `scripts/run_autoreply_parallel_draft_proxy.py`: resumable container,
  smoke, screen, metrics, and capacity orchestration.
- Create `tests/test_autoreply_parallel_draft_proxy_runner.py`: command rendering,
  pass/fail, resume, and invalidation tests with fake process results.
- Create
  `bench-runs/autoreply-parallel-draft-proxy-20260816/manifest/experiment.json`:
  immutable workload and environment manifest.
- Create
  `bench-runs/autoreply-parallel-draft-proxy-20260816/proxies/<method>/`:
  local generated configs, weights, and tensor manifests; do not add large
  weights to git.
- Create `bench-runs/autoreply-parallel-draft-proxy-20260816/results.jsonl`:
  append-only normalized candidate rows.
- Create `bench-runs/autoreply-parallel-draft-proxy-20260816/SUMMARY.md` and
  `result-summary.json`: compact final decision report.

## Task 1: Add pure proxy specifications and tests

**Files:**

- Create: `tests/test_parallel_draft_proxy.py`
- Create: `scripts/build_parallel_draft_proxy.py`

### Steps

- [x] Write tests that import the builder by file path and assert the fixed
  target dimensions, acceptance schedules, and method configs:

```python
def test_acceptance_schedules_have_requested_means():
    for k in (3, 7):
        schedules = proxy.acceptance_schedules(k)
        assert set(schedules) == {50, 55, 60}
        for target, rates in schedules.items():
            assert len(rates) == k
            assert rates == sorted(rates, reverse=True)
            assert sum(rates) / k == pytest.approx(target / 100)


def test_eagle3_proxy_uses_three_aux_hidden_states():
    spec = proxy.proxy_spec("eagle3")
    assert spec.num_speculative_tokens == 3
    assert spec.config["architectures"] == ["Eagle3LlamaForCausalLM"]
    assert spec.config["num_aux_hidden_states"] == 3
    assert spec.tensor_shapes["fc.weight"] == (5120, 15360)


def test_dspark_keeps_native_block_size():
    spec = proxy.proxy_spec("dspark")
    assert spec.num_speculative_tokens == 7
    assert spec.config["dspark_block_size"] == 7
    assert spec.config["n_predict"] == 7
```

- [x] Run the focused tests and require an import failure before implementation:

```bash
cd /root/llm-inference-benchmarking
.venv/bin/pytest -q tests/test_parallel_draft_proxy.py
```

Expected: FAIL because `scripts/build_parallel_draft_proxy.py` does not exist.

- [x] Implement immutable target constants, `ProxySpec`, three method specs,
  and exact schedules. The public pure API is:

```python
@dataclass(frozen=True)
class ProxySpec:
    method: str
    config: dict[str, object]
    tensor_shapes: dict[str, tuple[int, ...]]
    num_speculative_tokens: int


def acceptance_schedules(k: int) -> dict[int, list[float]]:
    if k == 3:
        return {
            50: [0.80, 0.50, 0.20],
            55: [0.85, 0.55, 0.25],
            60: [0.90, 0.60, 0.30],
        }
    if k == 7:
        return {
            50: [0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20],
            55: [0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25],
            60: [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30],
        }
    raise ValueError(f"unsupported speculative length: {k}")
```

- [x] Make `--method {eagle3,dflash,dspark} --dry-run --output DIR` write only
  `config.json` and `tensor-manifest.json`, atomically.

- [x] Run focused tests; expected PASS.

- [x] Commit the pure specification and tests:

```bash
git add scripts/build_parallel_draft_proxy.py tests/test_parallel_draft_proxy.py
git commit -m "test: define parallel draft proxy specifications"
```

## Task 2: Generate structurally valid proxy checkpoints

**Files:**

- Modify: `scripts/build_parallel_draft_proxy.py`
- Modify: `tests/test_parallel_draft_proxy.py`
- Create locally: `bench-runs/autoreply-parallel-draft-proxy-20260816/proxies/`

### Steps

- [ ] Add failing tests for tensor-source rules. Large vocabulary embeddings and
  compatible transformer weights must reuse the classic-EAGLE donor; new
  method-specific tensors use deterministic zero initialization. Each manifest
  entry records `name`, `shape`, `dtype`, `source`, and `nbytes`.

```python
def test_manifest_accounts_for_every_tensor_byte(tmp_path):
    manifest = proxy.build_manifest(proxy.proxy_spec("eagle3"), tmp_path)
    assert manifest["total_nbytes"] == sum(t["nbytes"] for t in manifest["tensors"])
    assert {t["source"] for t in manifest["tensors"]} <= {
        "classic-eagle-donor", "zero-init", "identity-init"
    }
```

- [ ] Run the focused tests; expected FAIL because `build_manifest` is absent.

- [ ] Implement safetensors header inspection without loading donor tensors,
  shape validation, deterministic tensor generation, temporary-file output,
  `fsync`, and atomic rename. Never overwrite a nonempty checkpoint unless
  `--force` targets that exact method directory.

- [ ] Keep optional shared embeddings/LM heads out of DFlash/DSpark only when
  the native loader explicitly aliases them from the target. EAGLE3 includes
  every tensor its loader does not skip.

- [ ] Run the builder inside the exact vLLM container so PyTorch and
  safetensors versions match runtime:

```bash
docker run --rm --gpus all --ipc=host \
  -v /root/llm-inference-benchmarking:/work \
  -w /work --entrypoint python3 vllm/vllm-openai:v0.27.1 \
  scripts/build_parallel_draft_proxy.py \
  --all \
  --donor bench-runs/autoreply-eagle-smoke-20260815/draft \
  --output bench-runs/autoreply-parallel-draft-proxy-20260816/proxies
```

Expected: three `config.json`, `model.safetensors`, and
`tensor-manifest.json` sets; process exits zero.

- [ ] Validate every safetensors header in a fresh container and require all
  byte ranges to end exactly at file size.

- [ ] Commit only the builder, tests, and compact tensor manifests; exclude
  generated `model.safetensors` files.

## Task 3: Add a resumable experiment controller

**Files:**

- Create: `tests/test_autoreply_parallel_draft_proxy_runner.py`
- Create: `scripts/run_autoreply_parallel_draft_proxy.py`

### Steps

- [ ] Write failing tests for command rendering and invariants:

```python
def test_server_command_preserves_fixed_contract():
    cmd = runner.server_command(candidate("eagle3", "bf16", 60))
    joined = " ".join(cmd)
    assert "--max-model-len 8192" in joined
    assert "--quantization modelopt" in joined
    assert '"method":"eagle3"' in joined
    assert '"rejection_sample_method":"synthetic"' in joined


def test_client_command_preserves_sampling():
    cmd = runner.client_command(qps=8.8, duration=30, output="result.json")
    joined = " ".join(cmd)
    for fragment in (
        "--max-tokens 50", "--temperature 0.7", "--top-p 0.8",
        "--top-k -1", "--frequency-penalty 0.01",
        "--presence-penalty 0.01",
    ):
        assert fragment in joined
    assert '"n":3' in joined
    assert '"stop":"<|im_end|>"' in joined
```

- [ ] Add tests proving an existing successful result is skipped, a failed
  candidate remains recorded, a prefix rate outside `[0.65, 0.68]` invalidates
  the point, and only controller-owned container names can be stopped.

- [ ] Run focused tests; expected FAIL before the controller exists.

- [ ] Implement `Candidate`, JSONL result schema, exact server/client command
  rendering, readiness polling, logs, Prometheus snapshots, counter deltas,
  output-drift extraction, and append-only resume behavior.

- [ ] Controller states are `PENDING`, `STARTING`, `SMOKE`, `SCREENING`,
  `BOUNDARY`, `PASS`, `FAIL`, and `UNSUPPORTED`. A process/container failure
  always becomes a durable result row before moving on.

- [ ] Implement container ownership guard:

```python
OWNED_PREFIX = "autoreply-proxy-"


def require_owned_container(name: str) -> None:
    if not name.startswith(OWNED_PREFIX):
        raise ValueError(f"refusing to manage non-benchmark container: {name}")
```

- [ ] Run focused tests; expected PASS. Run the existing replay tests touched by
  shared imports; expected PASS.

- [ ] Commit controller and tests.

## Task 4: Capture preflight and run native smoke matrix

**Files:**

- Create:
  `bench-runs/autoreply-parallel-draft-proxy-20260816/manifest/experiment.json`
- Append: `bench-runs/autoreply-parallel-draft-proxy-20260816/results.jsonl`
- Create per-candidate raw artifact directories.

### Steps

- [ ] Record Docker image digest, vLLM version, GPU/driver, disk, topology,
  target/dataset hashes, proxy hashes, exact help, and fixed request contract.

- [ ] Stop only the current benchmark-owned `autoreply-ngram-cpu7-boundary`
  container and verify the RTX 5090 has no other compute process.

- [ ] For each method, launch BF16 at synthetic 60%, run 1-QPS warmup/smoke,
  and validate service, draft counters, request fields, acceptance tolerance,
  graph/backend, and prefix-cache alignment.

- [ ] For each successful BF16 method, repeat with draft
  `quantization="fp8"`; if unsupported, try the exact v0.27.1 documented online
  alias `fp8_per_tensor` once and preserve both failures.

- [ ] Classify failures from full logs. Make at most one minimal native-config
  correction per method. Do not patch framework code in this task.

- [ ] Save a smoke matrix and identify runnable method/precision pairs.

## Task 5: Run acceptance and quantization screens

**Files:**

- Append: `bench-runs/autoreply-parallel-draft-proxy-20260816/results.jsonl`
- Create raw artifact directories under `<method>/<precision>/accept-{50,55,60}`.

### Steps

- [ ] For every runnable BF16 pair, cold-launch 50%, 55%, and 60%, warm at 2
  QPS for ten seconds, then screen at 8.5 QPS and 9.0 QPS for 30 seconds each.

- [ ] Repeat the same matrix for runnable FP8 pairs. Preserve target ModelOpt
  NVFP4 and target FP8 KV; only draft weight quantization changes. DFlash draft
  KV may remain BF16 when required and must be recorded.

- [ ] At each point require 100% success, acceptance within two points, prefix
  cache within 65%-68% after warm state, nonzero output, and nonzero draft work.

- [ ] Calculate BF16-to-FP8 p50, GPU-memory, draft-memory, and accepted-token
  efficiency deltas at identical load and acceptance.

- [ ] Stop dominated candidates according to the approved 5% rule and write the
  reason into results JSONL.

## Task 6: Establish capacity boundaries

**Files:**

- Append normalized probe/confirmation rows to `results.jsonl`.
- Create `boundary/` and `confirmations/` under each competitive candidate.

### Steps

- [ ] Search on a 0.1-QPS grid, beginning with a known passing low point before
  probing high. Use the fixed full dataset pool because the 1,000-row file
  cannot sustain the generic `0.02*qps` sample fraction for 30 seconds; record
  this established AutoReply exception.

- [ ] For every winning method/precision at 50%, 55%, and 60%, identify the
  largest point with 100% success and p50 E2E `<2s`.

- [ ] Run three fresh 30-second confirmations at the claimed boundary and one
  adjacent 0.1-QPS point. A winner is valid only if all three confirmations
  pass and the adjacent point fails.

- [ ] Leave the overall best safe candidate running on port 8080; if every
  proxy is unsafe, restore and leave the established no-speculative winner.

## Task 7: Optional ModelOpt NVFP4 diagnostic

**Files:**

- Modify only if supported: `scripts/build_parallel_draft_proxy.py`
- Add matching tests and local proxy artifacts.

### Steps

- [ ] Inspect ModelOpt FP4 serialization requirements and confirm all material
  draft linear layers have supported ModelOpt methods on SM120.

- [ ] If supported, first add a failing manifest test requiring FP4 weights and
  scales for every material linear layer; implement the minimal conversion and
  rerun the test.

- [ ] Smoke one 60%-acceptance candidate and prove from logs that ModelOpt FP4
  kernels execute. If any material layer falls back, record NVFP4 as unsupported
  and stop this task.

- [ ] If valid, compare it with BF16 and FP8 at the same method, acceptance, and
  QPS; only then consider one boundary search.

## Task 8: Produce and verify the training decision

**Files:**

- Create: `bench-runs/autoreply-parallel-draft-proxy-20260816/SUMMARY.md`
- Create: `bench-runs/autoreply-parallel-draft-proxy-20260816/result-summary.json`
- Modify: this plan's checkboxes as tasks complete.

### Steps

- [ ] Aggregate result rows and Prometheus deltas. Report method, K, precision,
  target/measured acceptance, mean accepted length, QPS boundary, p50s, target
  cycles, draft cost, output drift, prefix hit, GPU memory, and failure reason.

- [ ] Calculate draft-quantization uplift at equal conditions and interpolate
  the minimum acceptance needed to reach 9.5 QPS only when measured points
  bracket that value. Otherwise report a one-sided bound instead of an
  unsupported extrapolation.

- [ ] State `train`, `conditional train`, or `do not train` separately for
  EAGLE3, DFlash, and DSpark, including the required precision and acceptance.

- [ ] Invoke `verification-before-completion`. Validate JSON, report arithmetic,
  original request invariants, container command, health, artifact paths, and
  `git diff --check` with fresh commands.

- [ ] Commit compact source, tests, specs, plans, manifests, and reports using
  the configured Saddss identity. Do not commit proxy weights or bulky raw logs.

- [ ] Attempt to push `main` only if GitHub authentication reports Saddss with
  write access; otherwise preserve the local commit and report the exact
  credential blocker without changing accounts.
