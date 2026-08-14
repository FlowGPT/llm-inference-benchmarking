# AutoReply M12 production-alignment benchmark

Date: 2026-08-13 UTC

## Result

Max QPS meeting p50 client E2E `< 2.0s`: **9.3 QPS**

Method: offload OFF, vLLM `v0.27.1`, 12 continuous 30-second windows,
`auto-steady` with tail-6 fallback, precision 0.1, one HTTP request per QPS
unit, `n=3` completions per request.

| QPS | Verdict | Primary p50 | Tail-6 p50 | Prefix token hit rate | Evidence |
|---:|:---:|---:|---:|---:|---|
| 7.5 | PASS | 0.559s (steady-12) | 0.551s | 66.62% | `bench-runs/autoreply/qps_7.5_20260813_165124` |
| 9.3 | PASS | 1.709s (steady-3) | 1.761s | 66.75% | `bench-runs/autoreply/qps_9.3_20260813_170446` |
| 9.4 | FAIL | 2.085s (tail-6, no steady) | 2.085s | 66.45% | `bench-runs/autoreply/qps_9.4_20260813_173015` |
| 9.6 | FAIL | 3.720s (steady-12) | 3.771s | 66.45% | `bench-runs/autoreply/qps_9.6_20260813_172404` |
| 9.8 | FAIL | 8.320s (steady-8) | 8.835s | 66.45% | `bench-runs/autoreply/qps_9.8_20260813_171748` |
| 10.2 | FAIL | 20.205s (steady-6) | 20.205s | 66.45% | `bench-runs/autoreply/qps_10.2_20260813_171106` |
| 11.2 | FAIL | 48.752s (steady-6) | 48.752s | 66.45% | `bench-runs/autoreply/qps_11.2_20260813_165734` |
| 15.0 | FAIL | not enough rounds (9/12) | n/a | 66.45% | `bench-runs/autoreply/qps_15_20260813_164038` |

At 15 QPS, the nine observed window p50s were 12.24, 28.71, 45.52,
62.68, 116.29, 303.44, 321.85, 325.45, and 329.04 seconds; the final three
windows had no completed requests before the 300-second drain timeout.

## Alignment evidence

- `n=3`: non-streaming returned three choices with indices 0, 1, 2;
  streaming observed all three indices and their terminal finish reasons.
- Prefix caching: production target is 66%-67%. Every decisive probe measured
  roughly 66.45%-66.75% using counter deltas from
  `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total`.
- Distribution dataset verification:
  `datasets/autoreply_prod_verify.json` is PASS for 1000 exact-quota rows and
  310 boundary rows. All canonical prompts and first 16-token blocks are
  unique, so cache hits come from `n=3` shared prefill rather than artificial
  cross-request duplication.
- Boundary smoke: 7001-, 8121-, 3741-, and 3744-token samples completed at
  100% success without HTTP 400; evidence is
  `bench-runs/autoreply/boundary_smoke_20260813_173636`.
- Service readiness: authenticated `/v1/models` returned HTTP 200; the model
  advertises `max_model_len=8192`.

The local result is **9.3 external HTTP requests/s**. Because every request uses
`n=3`, vLLM processes about `9.3 * 3 = 27.9` generated sequences/s. Therefore,
if the stated online ~28-QPS metric is collected inside vLLM at the sequence or
completion layer, the two results are already aligned; it does not imply three
replicas. Confirm the online metric definition first. Only if ~28 means gateway
HTTP requests/s should replica aggregation, hardware, token distribution,
model artifact, and launch flags be investigated as explanations for a real
gap.
This benchmark used one RTX 5090 (32 GB); its preflight also reported a
virtualized PCIe Gen1 link, although this non-offload workload did not transfer
KV data over PCIe during steady inference.

## Invalid exploratory runs

The earlier `qps_15_20260813_162602` run exhausted a 1000-row input before the
360-second window. `qps_15_20260813_163219` used bucket-grouped ordering. Both
were stopped and excluded from the result table. The final route uses 13
complete shuffled cycles (`datasets/autoreply_prod_dist_repeated_13x.jsonl`).
