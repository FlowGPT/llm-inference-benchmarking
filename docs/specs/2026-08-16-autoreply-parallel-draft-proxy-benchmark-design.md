# AutoReply parallel-draft proxy benchmark design

Date: 2026-08-16

## Objective

Determine whether it is worth training a speculative draft model for the
fixed AutoReply deployment by measuring the real serving cost of EAGLE3,
DFlash, and DSpark under synthetic 50%, 55%, and 60% draft-token acceptance,
and by quantifying the benefit of draft-model quantization.

The experiment is a runtime-cost proxy. It does not claim that an untrained
draft can achieve the configured acceptance rate or preserve output quality.

## Fixed production contract

- Target checkpoint:
  `/root/.cache/huggingface/saved_models_user-model-afs-sfw-m12-v7_1s1600_nvfp4_kv_fp8`
- Served model name:
  `kaonai/user-model-afs-sfw-m12-v7.1s1600-nvfp4`
- Framework image: `vllm/vllm-openai:v0.27.1`
- Hardware: one NVIDIA GeForce RTX 5090
- Dataset: `/mnt/shared/sss/data/auto-reply-test.json`
- `max-model-len=8192`; it must not change.
- Request sampling remains `n=3`, `max_tokens=50`, temperature `0.7`, top-p
  `0.8`, top-k `-1`, min-p disabled, frequency penalty `0.01`, presence
  penalty `0.01`, and stop string `<|im_end|>`.
- Capacity gate: 100% HTTP success and client p50 E2E latency strictly below
  two seconds.
- The established no-speculative capacity reference is 9.5 HTTP QPS.
- Prefix-cache hit rate should settle near 66%-67%; a materially different
  value invalidates workload alignment until explained.

## Proxy-model principle

Create structurally valid draft checkpoints with initialized placeholder
weights. Each proxy must execute the real framework path for its method:

- draft layers, projections, attention, LM head or shared head;
- proposer scheduling and sampling;
- draft KV-cache allocation and updates;
- target multi-token verification;
- CUDA graphs and framework bookkeeping.

Only the rejection decision is synthetic. vLLM's native
`rejection_sample_method=synthetic` replaces the target/draft probability test
with configured random acceptance while leaving the preceding computation in
place.

Weight values do not establish quality. They affect generated token identities,
so accepted placeholder predictions may change response content, output length,
and stop frequency. Every performance result must therefore report output-token
length and finish-reason drift and be labelled as a cost proxy.

## Method-specific proxies

### EAGLE3

- Use the vLLM EAGLE3 Llama-compatible draft implementation, which is the
  closest native path for the Mistral target's inherited Llama interfaces.
- Use one draft transformer layer and the target dimensions: hidden size 5120,
  intermediate size 14336, vocabulary size 131072, 32 attention heads, and
  eight KV heads.
- Add the EAGLE3 auxiliary-hidden-state fusion tensors required by the runtime;
  the existing classic-EAGLE checkpoint cannot merely be relabelled.
- Use three speculative tokens. This entails the real recursive EAGLE3 draft
  work for each verification cycle.

### DFlash

- Use vLLM's native DFlash draft model and parallel-drafting proposer.
- Build a self-contained checkpoint with target-compatible hidden and vocabulary
  dimensions, a valid mask token, auxiliary target-layer selection, and all
  required projection and normalization weights.
- Keep the algorithm's native masked parallel block behavior.
- Prefer the smallest native block that exercises the parallel path and permits
  comparison with the workload. Do not call a causal or bypass implementation
  DFlash merely to make it launch.
- Override the draft KV dtype to BF16 when the non-causal attention backend
  rejects FP8 KV. Target KV remains FP8.

### DSpark

- Use vLLM's native self-contained DSpark model path when compatible with the
  Mistral target.
- Include the DFlash-style backbone, Markov correction head, required mappings,
  and block metadata.
- Use the checkpoint's native block size, initially seven tokens. vLLM forbids
  a speculative length smaller than the DSpark block because it produces an
  unsupported layout and incorrect output.
- If the unmodified vLLM 0.27.1 path rejects the Mistral target, record the
  exact incompatibility. A minimal compatibility patch may be evaluated only
  as a separately labelled diagnostic; patched results cannot silently become
  the production recommendation.

## Synthetic acceptance schedules

For three-token EAGLE3 or a DFlash configuration that natively uses three
positions, use unconditional per-position schedules:

| Target overall acceptance | Per-position unconditional rates |
| --- | --- |
| 50% | `[0.80, 0.50, 0.20]` |
| 55% | `[0.85, 0.55, 0.25]` |
| 60% | `[0.90, 0.60, 0.30]` |

For a seven-token native block, use:

| Target overall acceptance | Per-position unconditional rates |
| --- | --- |
| 50% | `[0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20]` |
| 55% | `[0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25]` |
| 60% | `[0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30]` |

Each list is non-increasing and its arithmetic mean equals the named overall
token acceptance. vLLM converts these unconditional values to conditional
probabilities for its early-terminating rejection loop.

A measured scenario is valid only when the aggregate Prometheus acceptance is
within two percentage points of its target and the per-position curve is
consistent with the configured schedule.

## Draft quantization study

The primary comparison is BF16 versus online FP8 weight/activation
quantization selected independently in `speculative_config.quantization`.
This preserves the target's ModelOpt NVFP4 quantization.

For each method that reaches a valid BF16 smoke test:

1. run an identical online-FP8 smoke test;
2. verify from startup logs and model inspection that draft layers, not the
   target, received the requested quantization configuration;
3. compare draft checkpoint/device footprint, startup memory, fixed-load p50
   E2E/TTFT/TPOT, and maximum stable QPS;
4. report the QPS delta at equal method, block size, acceptance schedule, and
   server configuration.

ModelOpt NVFP4 is optional. It becomes a formal candidate only if a supported
conversion produces serialized scales and the runtime uses ModelOpt FP4 kernels
for every material draft linear layer. A configuration that silently falls
back to BF16, dequantizes through a slower generic path, or skips unsupported
layers is recorded as unsupported rather than credited as FP4 performance.

INT8/INT4 paths are diagnostic only unless they use a hardware-appropriate
kernel on RTX 5090 and preserve the same model structure. The study must not
infer a training recommendation from a Marlin or bitsandbytes fallback whose
overhead dominates this small-batch draft workload.

Quantized placeholder weights still do not provide a quality estimate. The
quantization result answers only how much draft execution cost can change.

## Experiment stages

### Stage 1: preflight and checkpoint construction

- Record exact image, GPU, driver, source paths, free disk, and existing
  service state.
- Inspect adjacent vLLM model implementations and loader expectations before
  generating each checkpoint.
- Generate checkpoints deterministically and save a manifest of tensor names,
  shapes, dtypes, byte size, and initialization rule.
- Validate configs and safetensors without loading the target.

### Stage 2: method and quantization smoke tests

For each method, launch BF16 first and then online FP8 when BF16 succeeds.
Require:

- service readiness;
- one valid HTTP response path with the fixed request fields;
- nonzero draft and verification counters;
- synthetic acceptance within tolerance at low QPS;
- no silent proposer bypass;
- the expected model runner, attention backend, graph mode, and draft dtype in
  logs.

Failed launches remain in the result set with their exact exception and stage.

### Stage 3: bounded fixed-load screen

- Test 50%, 55%, and 60% for every BF16 method that passed smoke.
- Test the same schedules with FP8 when the FP8 method passed smoke.
- Start at a load below the known non-speculative boundary, then screen at the
  relevant 8.5-9.5 QPS band.
- Use a ten-second low-QPS warmup and 30-second fixed-QPS windows during
  candidate screening.
- Record client JSON, detailed request logs, server logs, Prometheus snapshots,
  acceptance, prefix-cache rate, output length, finish reasons, and GPU memory.

### Stage 4: capacity boundary

- Run 0.1-QPS boundary searches only for candidates that are competitive in
  Stage 3. Clearly dominated candidates do not receive full searches.
- Confirm each claimed winner with three independent 30-second runs at the
  passing boundary and one adjacent failing point.
- The final capacity statement uses HTTP QPS. Also state sampled sequence rate
  as `3 * HTTP QPS` because requests keep `n=3`.

### Stage 5: training decision

For every method, report:

- supported, patched diagnostic, or unsupported status;
- BF16 and quantized runtime cost;
- maximum stable QPS at each tested acceptance target;
- delta versus the 9.5-QPS no-spec baseline and versus classic EAGLE;
- measured accepted length and verification efficiency;
- minimum acceptance/maximum draft cost implied by break-even interpolation;
- output drift and why it prevents a quality conclusion;
- whether training is recommended, conditionally recommended, or rejected.

## Fairness and invalidation rules

- Never change the target checkpoint, request sampling, dataset,
  `max-model-len`, GPU count, or two-second SLO.
- Keep the established non-speculative winner's server parameters unless a
  method has a documented hard compatibility requirement. Every deviation must
  be isolated and disclosed.
- Restart the server between configurations so quantization, CUDA graphs, and
  cache state cannot leak across candidates.
- Do not compare a seven-token parallel block to a three-token recursive block
  using acceptance percentage alone. Report block size, expected accepted
  draft tokens, and committed tokens per verification cycle.
- Prefix-cache values outside the expected 66%-67% steady-state band require
  investigation before accepting the point.
- A server crash, request failure, zero draft counter, proposer bypass, or
  acceptance outside tolerance invalidates the performance point.
- Preserve all useful raw evidence, but commit only compact reports and
  manifests unless the user explicitly requests raw artifacts in git.

## Stopping rules

- Stop a method after its native unpatched path and one minimal compatibility
  attempt both fail for the same architectural reason.
- Stop a quantization path after logs prove the requested kernels are
  unsupported or two equivalent launch attempts fail for the same reason.
- Stop capacity refinement for a candidate that is at least 5% below another
  candidate at equal acceptance and precision with no compensating memory
  advantage relevant to this workload.
- Do not extrapolate synthetic results into a claim that training will achieve
  the assumed acceptance.

## Artifacts and lifecycle

Store results under:

`bench-runs/autoreply-parallel-draft-proxy-20260816/`

The root contains `SUMMARY.md`, a structured `result-summary.json`, a version
manifest, proxy checkpoint manifests, and subdirectories for each method,
precision, acceptance schedule, launch, and probe.

Only benchmark-owned containers may be stopped. The final winning service is
left running on port 8080 unless it is unsafe or no candidate completes. Raw
proxy checkpoints and large logs are local artifacts and are excluded from the
compact git commit.

## Success criteria

The study is complete when:

1. every method has a reproducible native result or a documented structural
   blocker;
2. every runnable method has 50%, 55%, and 60% synthetic acceptance evidence;
3. BF16 versus FP8 uplift is measured for every method whose FP8 path works;
4. competitive candidates have confirmed capacity boundaries;
5. the report states a numerical training go/no-go threshold without claiming
   synthetic output quality;
6. JSON, report arithmetic, live winner configuration, and artifact paths pass
   final verification.
