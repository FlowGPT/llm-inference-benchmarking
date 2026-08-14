# AutoReply Inference Framework Shootout Design

Date: 2026-08-14

## Objective

Find the highest reproducible external HTTP QPS for the AutoReply M12 workload
on one RTX 5090 while primary client p50 E2E latency remains strictly below
2.0 seconds. Independently tune vLLM and TensorRT-LLM, then compare their
champions under an interleaved cold-start protocol. Separately research, but do
not train, a compatible speculative draft model.

The production-aligned vLLM reference is 9.3 HTTP QPS at 1.709 seconds primary
p50 with a 66.75% prefix-cache token hit rate.

## Fixed invariants

- Exact target checkpoint, tokenizer, chat template, served model identity, and
  ModelOpt NVFP4 weight quantization.
- KV cache remains FP8 as declared by the checkpoint. Other KV dtypes may only
  be measured as labeled controls and cannot win the framework comparison.
- Exact dataset, request order, Chat Completions endpoint, and client timing
  semantics.
- Exact sampling contract: `n=3`, `max_tokens=50`, `temperature=0.7`,
  `top_p=0.8`, frequency and presence penalties `0.01`, disabled `min_p`,
  `top_k=-1`, and `stop=["<|im_end|>"]`.
- One RTX 5090 and no CPU/KV offload unless a fresh hardware health check
  explicitly permits it.
- Primary SLO: auto-steady client p50 E2E must be strictly less than 2.0
  seconds. Final QPS precision is 0.1.
- One QPS means one external HTTP request. Each request asks for three choices;
  engine-side sequence throughput must never be reported as HTTP QPS.

If a framework cannot load the exact checkpoint or cannot serve one request
with three choices and the fixed sampling contract, it is incompatible with
this workload. Client-side request splitting, weight conversion that changes
values or quantization, and silent sampling substitutions are forbidden.

## Frameworks

### vLLM

Use `vllm/vllm-openai:v0.27.1` and the approved design in
`docs/specs/2026-08-14-autoreply-vllm-qps-tuning-design.md`.

### TensorRT-LLM

Use the official stable release image
`nvcr.io/nvidia/tensorrt-llm/release:1.2.1` with `trtllm-serve` and the PyTorch
backend. Capture the image digest and complete `trtllm-serve --help` before
constructing candidates. If and only if 1.2.1 cannot load or correctly serve
the exact checkpoint, test `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22` as a
labeled compatibility fallback. Do not use the removed TensorRT engine backend
or an engine-build/checkpoint-conversion workflow.

## Isolation and ownership

Artifact root:
`/root/llm-inference-benchmarking/bench-runs/autoreply-framework-shootout-20260814`.

Only containers whose names begin with `autoreply-m12-vllm-` or
`autoreply-m12-trtllm-` belong to this experiment. Never stop or remove
`minio_test` or any unrelated container. Before each launch, pause if an
unrelated process occupies more than 2,048 MiB of GPU memory. Persist logs and
metrics before stopping a candidate. Do not store credentials in artifacts.

## TensorRT-LLM compatibility gate

Before performance testing:

1. Pull and pin the stable image digest; record TensorRT-LLM, PyTorch, CUDA,
   driver, GPU, and ModelOpt versions.
2. Capture the complete CLI help and resolved runtime configuration.
3. Load the exact local checkpoint using `trtllm-serve --backend pytorch`.
4. Verify health, `/v1/models`, and `/metrics`.
5. Verify non-streaming output has exactly choices 0, 1, and 2.
6. Verify streaming output uses the same three indices and valid finish reasons.
7. Verify every fixed request field is accepted rather than ignored or
   rewritten. Preserve request/response evidence with generated text redacted.
8. Verify finite outputs, no NaN/Inf, FP8 KV resolution, and stable memory.

Failure in stable 1.2.1 triggers exactly one equivalent gate on 1.3.0rc22. If
both fail, report TensorRT-LLM as incompatible rather than changing the model.

## TensorRT-LLM search

Inventory every startup option exposed by the selected image and assign a
disposition with a reason: fixed invariant, applicable search, applicable
control, not applicable, or unsupported/unsafe. Unknown flags require manual
review and cannot silently disappear.

Search in stages without a Cartesian product:

1. Establish the resolved-default baseline and a stable 9.4-QPS screen.
2. Screen one parameter family at a time, carrying forward only valid gains.
3. Test bounded interactions through coordinate descent.
4. Promote three materially distinct finalists to full QPS binary search.
5. Permit at most two adjacent refinements supported by a monotonic result.
6. Cold-confirm the winner and its adjacent `+0.1` QPS failure.

The parameter inventory must cover, when exposed and applicable:

- maximum batch size, sequence length, concurrency, and number of tokens;
- scheduler policy, capacity scheduler, overlap scheduler, batching waits, and
  chunked prefill;
- KV cache fraction/token capacity, FP8 dtype resolution, block size, block
  reuse, partial reuse, and prefix-aware scheduling;
- CUDA Graph enablement, padding, capture batch sizes, and eager controls;
- attention, GEMM/linear, sampler, compilation, and runtime kernel choices;
- HTTP frontend count, input-processing workers, postprocessing workers, and
  metrics collection interval;
- any target-only or no-training speculative mechanism as a labeled control,
  never as the no-speculation framework champion.

Aim for 24-36 launched TensorRT-LLM configurations, extending only while an
uncovered applicable family or reproducible adjacent improvement supplies a
concrete hypothesis. Preserve all startup failures, OOMs, request
incompatibilities, and regressions.

Screen at 9.4 HTTP QPS using six 30-second rounds and a three-round tail. Full
search uses 12 30-second rounds, tail six, LOW=8.5, HIGH=10.5, extrapolation,
and 0.1-QPS precision. Ranking and PASS/FAIL semantics match the vLLM design.

## Prefix-cache alignment

The expected 66%-67% value comes from within-request sharing among `n=3`
sequences, not conversation history or cross-request prompt reuse. For both
frameworks record:

- raw service counters and their documented units;
- client-computed prompt tokens and theoretical duplicated/shared tokens;
- a normalized token-level hit ratio using the same numerator and denominator.

A cold candidate must normalize to 66%-67%. A raw TensorRT-LLM metric may use
blocks, queries, or a different denominator; it is not compared directly until
normalized. Unexplained deviation invalidates the candidate.

## Cross-framework final

After each framework has a confirmed champion:

1. Rebuild both commands from durable manifests, not shell history.
2. Alternate framework order across three cold runs per champion at its claimed
   maximum QPS.
3. Require all three runs to pass the fixed gates and SLO.
4. Verify a valid failure at claimed QPS + 0.1 for each champion.
5. Select the highest all-run passing HTTP QPS. Break ties by median primary
   p50 across confirmations, then tail p50, error rate, variance, and finally
   configuration simplicity.

Report framework-internal sequence/token rates only as secondary metrics.

## Draft-model research

Research only; do not install a training stack, collect hidden states, train,
or benchmark speculative decoding. Evaluate NVIDIA Model Optimizer EAGLE3,
SpecForge EAGLE3, and other actively maintained Medusa/EAGLE/MTP alternatives
from primary documentation and source.

For each candidate answer:

- support for this custom 40-layer `MistralForCausalLM`, hidden size 5120,
  vocabulary 131072, and its chat template;
- whether the frozen target may be NVFP4 during training or a BF16 source
  checkpoint is required;
- online versus offline hidden-state training paths and estimated GPU, storage,
  data, and engineering requirements;
- checkpoint/export compatibility with vLLM, TensorRT-LLM, and SGLang;
- supported speculative algorithm and runtime constraints under non-greedy
  `n=3` sampling;
- maturity, maintenance, licensing, major risks, and a minimal future proof of
  concept.

The report recommends one primary and one fallback framework but makes no
performance claim without training and measuring acceptance rate.

## Deliverables

1. Complete per-image CLI compatibility matrices and manifests.
2. Every candidate, failure, exclusion, raw probe, cache snapshot, and log.
3. Confirmed vLLM and TensorRT-LLM maximum-QPS boundaries.
4. Cross-framework final table and exact redacted winning command.
5. Absolute and percentage improvement over the 9.3-QPS reference.
6. Prefix-cache normalization explanation and evidence.
7. Draft-model training-framework research report with primary sources.

Completion requires reproducible evidence for the selected winner, not merely
a promising screening result.
