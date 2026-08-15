# AutoReply serving benchmark report

Date: 2026-08-15

## Result

The production-safe winner remains vLLM at 9.4 HTTP requests/s. SGLang reaches
8.1 requests/s after launch-only tuning, while TensorRT-LLM reaches 5.1
requests/s. Every request uses `n=3`; these are not generated-sequence QPS
figures.

| Framework | Sustainable QPS | Adjacent failure or instability | Tail p50 evidence |
| --- | ---: | ---: | --- |
| vLLM | 9.4 | 9.5 | 1.808 / 1.879 / 1.871s confirmations |
| SGLang | 8.1 | 8.2 | 1.691 / 1.710 / 1.702s confirmations |
| TensorRT-LLM | 5.1 | 5.2 | 1.759 / 1.758 / 1.857s confirmations |

SGLang improves from a 6.5-QPS default long-run boundary to 8.1 QPS: +1.6
requests/s or +24.62%. It is 13.83% below vLLM and 58.82% above
TensorRT-LLM on this exact workload.

## Fixed protocol

- GPU: one NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.173.02.
- Model: `kaonai/user-model-afs-sfw-m12-v7.1s1600-nvfp4`.
- Model architecture: `MistralForCausalLM`; ModelOpt NVFP4 weights and FP8 KV.
- Context length: 8192 for every framework and candidate.
- Sampling: `n=3`, `max_tokens=50`, temperature 0.7, top-p 0.8, top-k -1,
  frequency/presence penalty 0.01, min-p disabled, stop `<|im_end|>`.
- SLO: tail-window mean of per-round client p50 E2E latency below 2 seconds.
- Formal runs: 12 rounds of 30 seconds with a six-round tail window and a cold
  server for each confirmation.

## SGLang compatibility finding

Official stable image `lmsysorg/sglang:v0.5.17` was used, pinned to digest
`sha256:16aba8925507e631e1dc1e23d95d026533602591775f6a8db68b74ee99746155`.

The legacy SGLang Mistral weight loader accepted this checkpoint without a load
error but returned gibberish with every tested attention/GEMM and KV dtype
variant. The exact checkpoint and requests were coherent in vLLM. Enabling
`SGLANG_ENABLE_WEIGHT_LOADER_V2=1` made both short and production-shaped
prompts coherent in stable SGLang. Performance results without this correctness
gate are invalid and were excluded.

## SGLang final launch

```bash
docker run --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e SGLANG_ENABLE_WEIGHT_LOADER_V2=1 \
  -p 30000:30000 \
  -v /path/to/model:/model:ro \
  --entrypoint python3 lmsysorg/sglang:v0.5.17 \
  -m sglang.launch_server \
  --model-path /model \
  --served-model-name kaonai/user-model-afs-sfw-m12-v7.1s1600-nvfp4 \
  --host 0.0.0.0 --port 30000 \
  --context-length 8192 \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.84 \
  --schedule-policy lpm \
  --mm-feature-transport cpu \
  --fp4-gemm-backend flashinfer_cutlass \
  --enable-metrics \
  --cuda-graph-max-bs 144 \
  --page-size 16 \
  --stream-interval 5
```

The dominant improvement was extending decode CUDA Graph coverage from the
default maximum batch 24 to 144. At 7 QPS this reduced p50 from 2.976s (fail)
to 0.866s (pass). Page size 16, memory fraction 0.84, and LPM scheduling added
headroom. Page sizes 32/64, mixed chunking, chunk sizes 1024/4096, and hard
running-request caps caused overload or regression. Graph maximum 160 was
indistinguishable from 144; 192 was slightly worse.

The 8.2-QPS candidate passed one 12-round run at 1.964s but failed another at
2.809s, including a 6.402s final round. It is therefore an unstable edge, not a
production capacity claim. Three cold 8.1-QPS runs passed at 1.691s, 1.710s,
and 1.702s.

## Why prefix-cache hit rate is 66-67%

This workload has no multi-turn history, but each HTTP request asks for three
samples. vLLM counted 16,723,773 queried prompt tokens and 11,112,608 cached
tokens. The query count is exactly divisible by three and the hit count by two:
one branch establishes the prompt KV and two sibling branches reuse it. The
ideal ratio is therefore 2/3. Cache-block alignment leaves a small uncached
tail, producing the observed 66.448% instead of exactly 66.667%.

SGLang's exported `sglang:cache_hit_rate` is an instantaneous per-scheduler-
batch gauge that oscillates with chunk and clone batches. A single scrape, or
an unnormalized sum of SGLang prefill logs, is not comparable with vLLM's
cumulative token counters.

## Draft-model training research

No draft model was trained or benchmarked. The most direct option for this
Mistral family is NVIDIA Model Optimizer EAGLE3, whose official speculative
decoding example lists Mistral support and SGLang/TensorRT-LLM export. SpecForge
is the preferred SGLang-native alternative and supports EAGLE3 training plus
online/offline/FSDP workflows, but this custom Mistral + NVFP4 checkpoint has no
published ready-made recipe. Treat it as a compatibility project requiring a
BF16/frozen target or validated online feature capture, tokenizer/vocabulary
checks, export validation, and acceptance-rate benchmarking before any QPS
claim. The official EAGLE repository is a reference fallback but requires more
custom-model porting.

## Cleanup and artifacts

- TensorRT-LLM raw candidate artifacts and its 1.2.1 image were removed as
  requested; normalized TensorRT-LLM results remain in `summary.json`.
- The SGLang nightly diagnostic image was removed after stable v0.5.17 passed.
- All benchmark containers and stale active-probe markers were removed.
- Structured SGLang results are in `sglang/summary.json`; raw pass/fail evidence
  remains under `sglang/`.
