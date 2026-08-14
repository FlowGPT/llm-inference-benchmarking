# AutoReply vLLM Max-QPS Tuning Design

Date: 2026-08-14

## Objective

Find the highest sustainable external HTTP QPS for the AutoReply M12 workload
on one RTX 5090 while the primary p50 client E2E latency remains strictly below
2.0 seconds. Tune only vLLM server startup parameters.

The previous production-aligned baseline is 9.3 HTTP QPS with a 1.709-second
primary p50 and 66.75% prefix-cache token hit rate.

## Fixed invariants

The experiment must not change:

- Model weights or served model name.
- ModelOpt NVFP4 weight quantization. KV-cache dtype remains eligible because
  it is a vLLM startup parameter, but any alternative must pass startup,
  finite-output, request-shape, memory, and SLO validation.
- AutoReply workload distribution and ordering.
- Any request sampling parameter: `n=3`, `max_tokens=50`, `temperature=0.7`,
  `top_p=0.8`, frequency and presence penalties `0.01`, disabled `min_p`,
  `top_k=-1`, and `stop=["<|im_end|>"]`.
- Chat endpoint and client E2E latency measurement semantics.
- SLO: primary p50 E2E must be strictly less than 2.0 seconds.
- Final binary-search precision: 0.1 HTTP QPS.
- Prefix-cache alignment gate: a cold candidate should remain in the expected
  66%-67% range. A material deviation invalidates the candidate until explained.
- CPU/KV offload is not part of the default search because the recorded
  preflight reports a virtualized PCIe Gen1 link. It may be added only after a
  fresh health check permits it or the user supplies the skill's explicit
  `force=true` override; otherwise it is recorded as excluded by the hardware
  safety gate rather than silently ignored.

The benchmark QPS unit is one external HTTP request. vLLM internally expands
each request into three generation sequences because `n=3`; engine-side
sequence rate must be labeled separately as approximately `3 * HTTP QPS`.

## Environment and isolation

- GPU: one NVIDIA GeForce RTX 5090, 32,607 MiB.
- Framework image: `vllm/vllm-openai:v0.27.1`.
- Model path:
  `/root/.cache/huggingface/saved_models_user-model-afs-sfw-m12-v7_1s1600_nvfp4_kv_fp8`.
- Canonical shared dataset: `/mnt/shared/sss/data/auto-reply-test.json`.
- Local benchmark worktree: `/root/llm-inference-benchmarking`.
- Artifact root:
  `/root/llm-inference-benchmarking/bench-runs/autoreply-tuning-20260814`.

Only containers whose names begin with `autoreply-m12-tune-` may be stopped or
removed by this experiment. Do not touch `minio_test` or any container that
appears after the run begins but does not use that prefix. Check GPU ownership
before every candidate launch; if an unrelated process occupies more than
2,048 MiB, pause instead of terminating it.

Each candidate gets a fresh server process and cold KV cache. Stop the previous
candidate only after its metrics and logs are durable. Leave the final winning
container running and report its name unless the user asks for cleanup.

Do not persist API keys, Hugging Face tokens, or token-shaped credentials in
commands recorded in result artifacts.

## Preflight

Before GPU-backed testing:

1. Capture GPU, driver, Docker image, vLLM version, and image digest.
2. Capture the target image's complete vLLM server help. Treat it as the flag
   compatibility source of truth; exclude unsupported candidate flags.
3. Run the non-offload health check and preserve all warnings.
4. Validate the canonical 1000-row dataset.
5. Rebuild a local 13-cycle route for probes long enough to consume more than
   1,000 requests. This is a derived artifact and is not committed.
6. Verify one non-streaming response has exactly three choices and one
   streaming response uses choice indices 0, 1, and 2.
7. Confirm `/metrics` exposes token-query and cached-token prefix counters.

## Search strategy

Use a broad, staged Tier-3 search. Cover every performance-relevant vLLM 0.27.1
parameter family that applies to this single-GPU, dense, text-only workload,
including families absent from the original production command. Do not take a
full Cartesian product: use one-family screening, retain directional winners,
then test interactions through bounded coordinate descent.

### Stage 0: complete parameter inventory

Parse the target image's `--help=all` output and write a compatibility matrix
with one row for every exposed startup flag. Classify each flag as:

- `fixed-invariant`: changing it would alter the model, request sampling, or
  workload semantics;
- `applicable-search`: plausible performance effect for this workload;
- `applicable-control`: useful as a negative/control measurement but unlikely
  to win;
- `not-applicable`: multi-node, multi-GPU parallelism, MoE, LoRA, multimodal,
  pooling, tool parsing, TLS, or observability plumbing unrelated to serving
  performance;
- `unsafe/unsupported`: rejected by the actual image/model/hardware or blocked
  by the health policy.

Record a reason for every flag not searched. This guarantees broad coverage
without wasting GPU hours on parameters that cannot affect this deployment.
Use official vLLM 0.27.1 documentation/source to establish semantics; the
target image's parser remains the authority for availability.

### Stage 1: baseline and one-family screening

Start from the current baseline:

```text
max_num_seqs=96
max_num_batched_tokens=8192
gpu_memory_utilization=0.94
chunked_prefill=on
async_scheduling=on
compilation=-O3
```

Screen the following ordered families. Exact values are chosen after reading
the resolved defaults and valid ranges from `--help=all`; suggested starting
values below are bounded, not an instruction to pass unsupported values. A
later family may build on the current winner, but every combined change must be
named in the candidate record.

| Candidate family | Values or comparison | Hypothesis | Main risk |
|---|---|---|---|
| Sequence capacity | `64`, `96`, `128`, `160` | Find the best concurrency ceiling for three child sequences per HTTP request | Too low underfills; too high increases queueing or graph/KV pressure |
| Batched tokens | `8192`, `12288`, `16384` with the best sequence limit | Larger prefill batches may improve long-prompt throughput | Larger batches may hurt p50 or exceed graph/memory limits |
| Scheduled tokens and microbatching | Supported `max_num_scheduled_tokens` and `ubatch_size` variants | Separate scheduler admission from execution microbatch size | Poor values can add fragmentation or queueing |
| Chunked prefill | on versus off on the best capacity pair | Validate whether interleaving long prefills helps this short-decode workload | Disabling may cause head-of-line blocking |
| Async scheduling | on versus off on the best capacity pair | Measure host-scheduling benefit on a single GPU | Some combinations may regress latency or be unsupported |
| Prefill scheduler | `long_prefill_token_threshold`, `prefill_schedule_interval`, and `scheduler_reserve_full_isl` variants | Long 3k-4k prompts may benefit from different admission/interleave behavior | Can starve decode or reduce batch packing |
| GPU memory fraction | `0.94`, `0.95`, `0.96` on the current winner | More KV capacity may reduce pressure at the SLO boundary | Less graph/activation headroom or startup OOM |
| KV layout and precision | Supported `block_size`, `prefix_match_unit`, FP8-versus-auto KV dtype, and scale-calculation controls | Reduce block-tail waste or improve KV throughput/capacity | Numerical drift, lower cache capacity, or unsupported combinations |
| Prefix-sharing implementation | `kv_sharing_fast_prefill` off/on when the model advertises support | Reduce duplicate prefill work for `n=3` | No-op or unsafe when the model lacks sharing metadata |
| Attention and linear kernels | Supported attention/linear backends and FlashInfer autotune | A different kernel may better fit Blackwell and 3k-4k prefill | Import/startup failure or shape-specific regression |
| CUDA graph and compile | Optimization levels, performance mode, capture-size controls, and eager mode as a control | Reduce launch overhead and improve shape coverage | Capture memory bloat or reduced dynamic-shape performance |
| Allocator/runtime | `enable_cumem_allocator` and supported kernel/runtime config controls | Improve allocation stability and reduce runtime overhead | Startup incompatibility or no measurable effect |
| API/tokenization frontend | `api_server_count`, `renderer_num_workers`, access-log controls, and supported tokenizer/rendering worker settings | Remove CPU/API bottlenecks feeding short responses | Multiple frontends can add IPC overhead or duplicate memory |
| Streaming cadence | Supported `stream_interval` values with final-output equivalence checks | Reduce network/event-loop overhead from three short streams | Changes chunk timing and may affect perceived latency |
| Speculative decoding | Target-only/ngram methods that require no different model, if supported with `n=3` | Short decode may gain modest throughput without changing target distribution | Setup overhead may dominate 50-token outputs or violate compatibility |

Explicitly classify but normally exclude single-GPU-inapplicable TP/PP/DP/DCP,
expert parallel/MoE, LoRA, multimodal caches, model loading, TLS, tracing, and
debug-only flags. Test a control only when it can falsify a concrete bottleneck
hypothesis.

Target 18-28 Stage-1 launches and allow up to 36 total launched configurations
across all screening and interaction stages. A family may use fewer candidates
when the first supported alternative clearly regresses or fails. A family may
use one extra adjacent value when it shows a monotonic improvement. Preserve
every failure instead of retrying unsupported combinations indefinitely.
The 36-candidate value is a soft bound, not a time budget: extend it when an
uncovered applicable family or reproducible improvement still has a concrete
next hypothesis. Stop only when all applicable families have a disposition and
the remaining adjacent/interaction candidates no longer produce material gain.

Use a fixed 9.4 HTTP QPS screening probe because it is immediately above the
known 9.3-QPS baseline boundary. Screening uses six 30-second rounds with a
three-round tail and auto-steady analysis. These short probes rank candidates;
they are not final max-QPS claims.

Rank Stage-1 candidates by:

1. Valid workload and prefix-cache alignment.
2. SLO pass.
3. Lower primary p50 at 9.4 QPS.
4. Lower tail p50 and variance.
5. Fewer request errors and less queue growth.
6. Simpler configuration when performance is effectively tied.

Keep the best setting from each useful family for interaction testing. If every
candidate fails 9.4, rank valid candidates by primary p50 instead of declaring
that no tuning helped.

### Stage 2: interaction and coordinate-descent screening

Starting from the best Stage-1 configuration, add retained family winners one
at a time. Keep a change only when it improves primary p50 by at least 2%,
turns a valid 9.4-QPS FAIL into PASS, improves stability without reducing p50,
or unlocks a higher fixed-QPS screen. Re-test surprising improvements once from
a cold server before retaining them.

Test at least these interactions when their individual flags are supported:

- sequence capacity x batched/scheduled tokens;
- chunked prefill x prefill scheduler controls;
- GPU memory fraction x CUDA graph capture budget;
- block/prefix granularity x `n=3` prefix sharing;
- attention backend x compilation/performance mode;
- API frontend workers x async scheduling.

Promote the best three materially distinct configurations to full search. Avoid
three finalists that differ only by an adjacent scalar value.

### Stage 3: full max-QPS binary search

For each of the three finalists:

- Start with LOW=8.5 and HIGH=10.5 HTTP QPS.
- Probe LOW before HIGH.
- Use 12 rounds of 30 seconds, tail window 6, and auto-steady analysis.
- Snapshot prefix-cache counters immediately before and after every probe.
- Extrapolate upward if HIGH passes; extrapolate downward if LOW fails.
- Continue to 0.1-QPS precision.
- Treat incomplete rounds or server death under a load above a confirmed PASS
  as a load FAIL, preserve the evidence, relaunch the same candidate, and
  continue below it.

The official PASS/FAIL signal is the analyzer's primary auto-steady p50, with
the legacy tail-6 value always reported alongside it.

### Stage 4: bounded refinement and confirmation

If the full searches reveal one clear monotonic parameter direction or a strong
interaction not resolved by screening, allow up to two adjacent refinement
configurations. They must preserve every fixed invariant and use the same full
binary-search method.

Select the winner by highest confirmed HTTP QPS. Break a QPS tie using lower
primary p50 at that QPS, then lower tail p50, then simpler startup parameters.
Confirm the winner with one additional cold 12-round probe at its reported max
QPS. Ensure the adjacent 0.1-QPS point has a valid FAIL observation; run it if
the binary search did not already produce one.

## Measurements per candidate

Record:

- Exact redacted startup command and candidate diff from baseline.
- Startup success, readiness time, image and framework version.
- Resolved cache configuration: block size, KV token capacity, GPU blocks, and
  maximum concurrency.
- Per-round target/actual/completion QPS, p50 E2E, TTFT, TPOT, success rate,
  queue symptoms, and finish reasons.
- Primary p50 window, tail-6 p50, warmup-dominated flag, and analyzer notes.
- Prefix-cache hit/query token deltas and hit rate.
- Server/container liveness, OOM or startup errors, and relevant log tail.

Failed and excluded candidates remain in the result table with explicit
reasons. Do not select a faster candidate that violates the SLO, prefix gate,
request semantics, or model/sampling invariants.

## Output

Produce:

1. The Stage-0 complete flag compatibility matrix, including exclusion reasons.
2. Stage-1 one-family and Stage-2 interaction tables for every attempted
   candidate.
3. Per-probe binary-search tables for each finalist.
4. Baseline-versus-winner max-QPS delta in absolute QPS and percent.
5. The exact redacted winning launch command.
6. Artifact paths for help/version manifests, server logs, Prometheus
   snapshots, raw client JSONL, normalized candidate results, and summary.
7. The final winner container name and whether it remains running.

The result is successful only when a winning configuration has a cold
confirmation at its reported max QPS, a valid adjacent FAIL boundary, and
preserved evidence that all fixed invariants held.
