# AutoReply Simple CPU KV Offload 50 GiB

Date: 2026-08-16

## Verdict

`SimpleCPUOffloadConnector` with a 50 GiB native CPU KV tier does not help this
workload. The highest tested passing point is 9.0 HTTP QPS, versus the confirmed
9.5 HTTP QPS no-offload baseline: **-0.5 QPS / -5.26%**.

Pass requires 100% HTTP success and p50 end-to-end latency strictly below 2 s.

## Fixed contract

- Dataset: `/mnt/shared/sss/data/auto-reply-test.json`
- Dataset SHA-256: `501742f94ea6391c74fd624e291bb3cda472d2519a626f8d2a3813a0970eb652`
- Image: `vllm/vllm-openai:v0.27.1`
- Model: `saved_models_user-model-afs-sfw-m12-v7_1s1600_nvfp4_kv_fp8`
- Request: chat, `n=3`, `max_tokens=50`, temperature 0.7, top-p 0.8,
  top-k -1, min-p disabled, frequency/presence penalty 0.01, stop
  `<|im_end|>`
- `max-model-len=8192` was unchanged.
- Common server tuning: max-num-seqs 96, max-num-batched-tokens 8192,
  max-num-scheduled-tokens 3072, GPU memory utilization 0.94, async scheduling,
  prefix caching, chunked prefill, ModelOpt NVFP4, FP8 KV, O3.
- Experiment-only settings: `VLLM_USE_SIMPLE_KV_OFFLOAD=1`,
  `--kv-offloading-size 50`, `--kv-offloading-backend native`.

## Results

| Target QPS | HTTP success | p50 E2E | Verdict |
|---:|---:|---:|:---|
| 8.0 | 100% | 1.035 s | pass |
| 9.0 | 100% | 1.788 s | pass |
| 9.0 confirmation 2 | 100% | 1.852 s | pass |
| 9.0 confirmation 3 | 100% | 1.905 s | pass |
| 9.1 | 100% | 2.045 s | fail |
| 9.2 | 100% | 2.191 s | fail |
| 9.5 | 100% | 2.430 s | fail |

The no-offload 9.5-QPS confirmation p50 values were 1.972, 1.917, and 1.890 s.
At the same 9.5 QPS, offload raised p50 to 2.430 s, 26.8% above the no-offload
median confirmation.

## Cache evidence

Cold 8.0-QPS run:

- local prefix queries: 2,391,600 tokens
- local prefix hits: 1,589,216 tokens (66.45%)
- external connector queries: 802,384 tokens
- external connector hits: 0 tokens

The two final 9.0-QPS confirmations added 1,804,464 external queries and still
zero external hits. The local 66-67% hit rate is the expected `n=3` effect: the
first branch establishes the prompt KV and the other two branches reuse it.
There is no cross-request history reuse for the CPU tier to recover.

The connector also did not reduce the GPU KV allocation: startup still reported
19.3 GiB / 252,944 GPU KV tokens. It adds a lower cache tier; it does not turn
the 50 GiB into extra GPU-resident capacity. With no external hits, its cache
lookup/bookkeeping path is pure overhead near saturation.

## Offload health gate

- PCIe link: Gen5 x16, current equals maximum
- PCIe AER correctable/non-fatal/fatal errors: 0 / 0 / 0
- Pinned-memory bandwidth in the same image, 256 MiB transfers:
  - H2D median 6.097 ms, 44.03 GB/s
  - D2H median 5.272 ms, 50.92 GB/s

Therefore the regression is not attributable to a degraded PCIe link.

Raw JSON, request CSVs, Prometheus snapshots, health check output, and server
startup logs are retained under this directory.
