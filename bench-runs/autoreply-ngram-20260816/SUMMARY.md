# AutoReply n-gram speculative-decoding benchmark

Date: 2026-08-16

## Verdict

N-gram speculative decoding does **not** improve this workload on the tested
vLLM 0.27.1 / RTX 5090 stack. The best reproducible n-gram configuration is
the CPU proposer with one speculative token and an exact five-token lookup
window:

```text
--speculative-config '{"method":"ngram","num_speculative_tokens":1,"prompt_lookup_min":5,"prompt_lookup_max":5}'
```

Its stable p50-E2E-<2s boundary is **9.0 HTTP QPS**, versus **9.5 HTTP QPS**
without speculative decoding: **-0.5 QPS / -5.26%**. All HTTP requests retain
`n=3`, so 9.0 HTTP QPS creates about 27 sampled sequences per second.

## Fixed workload

- Model: `/root/.cache/huggingface/saved_models_user-model-afs-sfw-m12-v7_1s1600_nvfp4_kv_fp8`
- Image: `vllm/vllm-openai:v0.27.1`
- Dataset: `/mnt/shared/sss/data/auto-reply-test.json`
- GPU: one NVIDIA GeForce RTX 5090
- `max-model-len=8192` (never changed)
- `n=3`, `max_tokens=50`, temperature 0.7, top-p 0.8, top-k -1
- frequency penalty 0.01, presence penalty 0.01, min-p disabled
- stop string `<|im_end|>`
- Gate: 100% request success and client p50 E2E latency strictly below 2s
- Each measured window: fixed-QPS replay for 30 seconds after warmup

The general benchmark skill's `sample_end=0.02*qps` rule could not sustain a
30-second window with this 1,000-row dataset (for example, 1 QPS selected only
20 rows and produced 0.667 actual QPS). The established AutoReply protocol was
therefore retained: use the full sample pool, from which the client sends only
the requests required during each 30-second window.

## Candidate screen at 9.5 HTTP QPS

| Candidate | p50 E2E | Draft tokens | Accepted | Conditional acceptance | Mean output/HTTP | Result |
|---|---:|---:|---:|---:|---:|---|
| GPU n-gram, K=3, min=1, max=5 | 5.321s | 37,509 | 2,077 | 5.54% | 53 | fail |
| GPU n-gram, K=1, min=1, max=5 | 4.955s | 13,220 | 1,677 | 12.69% | 55 | fail |
| GPU n-gram, K=1, min=2, max=5 | 5.159s | 13,524 | 977 | 7.22% | 53 | fail |
| CPU n-gram, K=1, min=1, max=5 | 3.401s | 9,857 | 1,724 | 17.49% | 54 | fail |
| **CPU n-gram, K=1, min=5, max=5** | **2.266s** | 539 | 367 | 68.09% | 55 | **screen winner, fail at 9.5** |
| CPU n-gram, K=2, min=5, max=5 | 3.334s | 693 | 412 | 59.45% | 54 | fail |
| CPU n-gram, K=3, min=5, max=5 | 2.301s | 1,018 | 549 | 53.93% | 54 | fail |
| CPU K=1, min=max=5, FlashAttention + BF16 KV | 7.206s | 503 | 348 | 69.18% | 54 | fail |

Every measured candidate completed with 100% request success. Output length
remained 53-55 tokens per HTTP request, matching the non-speculative baseline
of about 54.5, so the result is not caused by synthetic acceptance or an
earlier-stop quality drift.

Two additional launch-only candidates were rejected before measurement:

- CPU `ngram` plus `--async-scheduling`: vLLM rejects this combination. CPU
  n-gram disables async scheduling.
- FlashAttention with FP8 KV on RTX 5090: vLLM rejects it because this path
  requires FA3 on SM90 or FA4 on SM100. Explicit BF16 KV launched and retained
  full CUDA graphs, but was much slower as shown above.

## Boundary and confirmations

| HTTP QPS | p50 E2E observations | Verdict |
|---:|---|---|
| 9.5 | 2.266s (winner screen) | fail |
| 9.3 | 2.124s | fail |
| 9.2 | probe 1.988s; confirmations 2.013 / 3.178 / 2.169s | unstable/fail |
| 9.1 | confirmations 1.983 / 1.958 / 2.029s | unstable/fail |
| **9.0** | probe 1.567s; confirmations **1.793 / 1.826 / 1.841s** | **stable pass** |

The three 9.0 confirmations all achieved 100% success. Their mean p50 was
1.820s. The non-speculative reference remains 9.5 QPS with three p50s of
1.972 / 1.917 / 1.890s; its adjacent 9.6-QPS check failed at 2.124s.

## Acceptance, coverage, and prefix cache

Across the three winning 9.0-QPS confirmations:

- output tokens: 44,218
- draft opportunities emitted: 1,575
- draft tokens: 1,575
- accepted draft tokens: 1,138
- conditional acceptance: **72.254%**
- estimated target verification cycles: `44,218 - 1,138 = 43,080`
- proposal coverage: `1,575 / 43,080 = 3.656%`
- accepted drafts as a share of output: `1,138 / 44,218 = 2.574%`
- effective tokens committed per target cycle: `44,218 / 43,080 = 1.0264`
- prefix-cache hit rate: **66.448%**

The 66.448% prefix hit is the expected `n=3` same-request branch reuse, not
multi-turn history reuse: the first child computes/populates the prompt and
the other two children reuse it. It aligns with the production observation of
66%-67%. Once the three sampled outputs diverge, their decode state cannot be
shared.

The apparently high 72.254% n-gram acceptance is therefore misleading in
isolation. Exact five-token matches occur on only about 3.66% of target cycles;
the other 96%+ receive no draft. The feature saves only about 2.57% of output
token steps before its runtime overheads are counted.

## Why it loses

1. **The workload is prefill dominated.** The preceding operator-level study
   attributes about 98.2% of useful FLOPs to prefill and only 1.8% to decode.
   N-gram changes decode only, so even free perfect decode has little room to
   raise end-to-end QPS.
2. **The generated suggestions are not extractive.** They are novel sampled
   next-user turns. Loose one-token lookup proposes frequently but accepts only
   12%-17% for K=1 (and 5.54% overall for GPU K=3). Exact five-token lookup is
   accurate when it fires, but fires too rarely.
3. **CPU n-gram loses baseline runtime optimizations.** vLLM disables async
   scheduling, uses the V1 runner, and FlashInfer forces speculative attention
   from `FULL_AND_PIECEWISE` CUDA graphs down to `PIECEWISE`.
4. **GPU n-gram preserves async scheduling but is expensive here.** With loose
   matching, GPU prompt scanning/proposal bookkeeping plus low-value target
   verification drove 9.5-QPS p50 to roughly 5 seconds.
5. **`n=3` strengthens the non-speculative baseline.** Two of three prompt
   branches reuse prefix KV, while n-gram lookup and verification must operate
   after the sampled branches diverge.
6. **The decode horizon is short.** About 54 output tokens per HTTP request is
   only about 18 tokens per child. There are too few useful decode steps over
   which to amortize the speculative runner and scheduler overhead.

The host health check reported a virtualized PCIe Gen1 x16 link rather than
Gen5 x16 and no torch installation for pinned-bandwidth measurement. This run
uses no offload, and the no-spec baseline was measured on the same host, so the
relative deployment verdict remains valid. CPU n-gram may additionally pay a
small latency penalty for host-originated proposals on this link; importantly,
the all-GPU n-gram implementation was still materially slower.

## Best n-gram launch command

```bash
docker run -d --rm --init --name autoreply-ngram-final \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8080:8080 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint python3 vllm/vllm-openai:v0.27.1 \
  -m vllm.entrypoints.openai.api_server \
  --model /root/.cache/huggingface/saved_models_user-model-afs-sfw-m12-v7_1s1600_nvfp4_kv_fp8 \
  --served-model-name kaonai/user-model-afs-sfw-m12-v7.1s1600-nvfp4 \
  --host 0.0.0.0 --port 8080 --dtype auto \
  --max-model-len 8192 \
  --enable-chunked-prefill --enable-prefix-caching \
  --max-num-seqs 96 --max-num-batched-tokens 8192 \
  --quantization modelopt --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.94 \
  --stream-interval 5 -O3 \
  --max-num-scheduled-tokens 3072 \
  --compilation-config '{"mode":3,"cudagraph_capture_sizes":[1,2,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81,84,87,90,93,96,99,102,105,108,111,114,117,120,123,126,129,132,135,138,141,144,147,150,153,156,159,162,165,168,171,174,177,180,183,186,189,192],"max_cudagraph_capture_size":192}' \
  --disable-uvicorn-access-log \
  --speculative-config '{"method":"ngram","num_speculative_tokens":1,"prompt_lookup_min":5,"prompt_lookup_max":5}'
```

This command is the best n-gram command, but it should **not** replace the
production no-spec command: the latter sustains 9.5 rather than 9.0 HTTP QPS.

## Artifact map

- `candidates/`: cold-start candidate logs, Prometheus snapshots, client JSON,
  and per-request CSV files.
- `final/cpu-k1-min5-max5/probes/`: boundary probes.
- `final/cpu-k1-min5-max5/confirmations/`: three-run confirmation sets for
  9.2, 9.1, and 9.0 QPS.
- `final/cpu-k1-min5-max5/container-inspect.json`: exact live winner container.
- `manifest/health-check.json`: preflight details including the PCIe warning.

The final n-gram container was intentionally left running as
`autoreply-ngram-final` on port 8080, following the benchmark lifecycle policy.
