# vLLM flag compatibility

| Flag | Class | Reason |
|---|---|---|
| `--additional-config` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--aggregate-engine-logging` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--all2all-backend` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--allow-credentials` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--allow-deprecated-quantization` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--allowed-headers` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--allowed-local-media-path` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--allowed-media-domains` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--allowed-methods` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--allowed-origins` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--api-key` | fixed-invariant | local authentication is fixed across candidates |
| `--api-server-count` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--async-scheduling` | applicable-search | host scheduling overhead can affect throughput |
| `--attention-backend` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--attention-config` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--block-size` | applicable-search | KV block granularity can affect waste and reuse |
| `--calculate-kv-scales` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--chat-template` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--chat-template-content-format` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--code-revision` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--cohere-format` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--cohere-is-reasoning-model` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--collect-detailed-traces` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--compilation-config` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--config` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--config-format` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--convert` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--cp-kv-cache-interleave-size` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--cpu-distributed-timeout-seconds` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--cpu-offload-gb` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--cpu-offload-params` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--cpunodebind` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--cudagraph-capture-sizes` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--cudagraph-metrics` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--data-` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--data-parallel-` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--data-parallel-address` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--data-parallel-backend` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--data-parallel-external-lb` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--data-parallel-hybrid-lb` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--data-parallel-multi-port-external-lb` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--data-parallel-rank` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--data-parallel-rpc-port` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--data-parallel-size` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--data-parallel-size-local` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--data-parallel-start-rank` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--data-parallel-supervisor-port` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--dbo-decode-token-threshold` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--dbo-prefill-token-threshold` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--dcp-comm-backend` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--dcp-kv-cache-interleave-size` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--decode-context-parallel-size` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--default-chat-template-kwargs` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--default-mm-loras` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--device-ids` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--diffusion-config` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--disable-access-log-for-endpoints` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--disable-cascade-attn` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--disable-chunked-mm-input` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--disable-custom-all-reduce` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--disable-fastapi-docs` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--disable-hybrid-kv-cache-manager` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--disable-log-` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--disable-log-stats` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--disable-nccl-for-dp-synchronization` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--disable-sliding-window` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--disable-uvicorn-access-log` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--distributed-executor-backend` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--distributed-timeout-seconds` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--download-dir` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--dp-supervisor-probe-failure-threshold` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--dp-supervisor-probe-interval-s` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--dp-supervisor-probe-timeout-s` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--dtype` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--ec-transfer-config` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--enable-auto-tool-choice` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--enable-bf16x3-router-gemm` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--enable-chunked-prefill` | applicable-search | prefill/decode interleaving can affect p50 |
| `--enable-cumem-allocator` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--enable-dbo` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--enable-elastic-ep` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--enable-ep-weight-filter` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--enable-eplb` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--enable-expert-parallel` | not-applicable | target is a dense Mistral model |
| `--enable-fault-tolerance` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--enable-flash-late-interaction` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--enable-flashinfer-autotune` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--enable-force-include-usage` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--enable-layerwise-nvtx-tracing` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--enable-log-deltas` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--enable-log-outputs` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--enable-log-requests` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--enable-logging-iteration-details` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--enable-lora` | not-applicable | workload does not use LoRA |
| `--enable-mamba-cache-stochastic-rounding` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--enable-mfu-metrics` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--enable-mixed-moe-lora-format` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--enable-mm-embeds` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--enable-moe-shared-loras` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--enable-offline-docs` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--enable-per-request-metrics` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--enable-prefix-caching` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--enable-prompt-embeds` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--enable-prompt-tokens-details` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--enable-request-id-headers` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--enable-return-routed-experts` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--enable-server-load-tracking` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--enable-sleep-mode` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--enable-ssl-refresh` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--enable-tokenizer-info-endpoint` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--enable-tower-connector-lora` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--enforce-eager` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--eplb-config` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--exclude-tools-when-tool-choice-none` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--expert-placement-strategy` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--fail-on-environ-validation` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--fault-tolerance-config` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--fingerprint-mode` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--fingerprint-value` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--fully-sharded-loras` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--gdn-prefill-backend` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--generation-` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--generation-config` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--gpu-memory-utilization` | applicable-search | KV capacity and graph headroom trade off |
| `--grpc` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--h11-max-header-count` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--h11-max-incomplete-event-size` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--headless` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--help` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--hf-config-path` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--hf-overrides` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--hf-token` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--host` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--ignore-patterns` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--interleave-mm-strings` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--io-processor-plugin` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--ir-op-priority` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--jit-monitor-mode` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--jit-monitor-verbose` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--json-arg` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--kda-prefill-backend` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--kernel-config` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--kv-cache-dtype` | applicable-search | control only; winning dtype remains FP8 |
| `--kv-cache-dtype-skip-layers` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--kv-cache-memory-bytes` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--kv-cache-metrics` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--kv-cache-metrics-sample` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--kv-events-config` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--kv-offloading-backend` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--kv-offloading-size` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--kv-sharing-fast-prefill` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--kv-transfer-config` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--language-model-only` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--limit-` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--limit-mm-per-prompt` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--linear-backend` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--load-format` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--log-config-file` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--log-error-stack` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--logits-processors` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--logprobs-mode` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--long-prefill-token-threshold` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--lora-dtype` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--lora-modules` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--lora-target-modules` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--mamba-backend` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--mamba-block-size` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--mamba-cache-dtype` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--mamba-cache-mode` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--mamba-cache-philox-rounds` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--mamba-ssm-cache-dtype` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--mamba-ssu-algorithm` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--master-addr` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--master-port` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--max-cpu-loras` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--max-cudagraph-capture-size` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--max-log-len` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--max-logprobs` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--max-lora-rank` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--max-loras` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--max-model-len` | fixed-invariant | workload context limit is immutable |
| `--max-num-batched-tokens` | applicable-search | batch token budget can affect packing |
| `--max-num-scheduled-tokens` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--max-num-seqs` | applicable-search | sequence admission capacity can affect throughput |
| `--max-parallel-loading-workers` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--media-io-kwargs` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--membind` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--middleware` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--mm-encoder-attn-backend` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--mm-encoder-attn-dtype` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--mm-encoder-fp8-scale-path` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--mm-encoder-fp8-scale-save-margin` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--mm-encoder-fp8-scale-save-path` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--mm-encoder-only` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--mm-encoder-tp-mode` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--mm-hasher-algorithm` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--mm-ipc-gpu-memory-gb` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--mm-processor-cache-gb` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--mm-processor-cache-type` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--mm-processor-kwargs` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--mm-shm-cache-max-object-size-mb` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--mm-tensor-ipc` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--model` | fixed-invariant | target checkpoint is immutable |
| `--model-class-overrides` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--model-impl` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--model-loader-extra-config` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--moe-backend` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--nnodes` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-allow-credentials` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-allow-deprecated-quantization` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-async-scheduling` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--no-calculate-kv-scales` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-cohere-is-reasoning-model` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-cudagraph-metrics` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--no-data-parallel-external-lb` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--no-data-parallel-hybrid-lb` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--no-disable-cascade-attn` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--no-disable-chunked-mm-input` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--no-disable-custom-all-reduce` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-disable-fastapi-docs` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-disable-hybrid-kv-cache-manager` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--no-disable-nccl-for-dp-synchronization` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-disable-sliding-window` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-disable-uvicorn-access-log` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--no-enable-auto-tool-choice` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-enable-bf16x3-router-gemm` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-enable-chunked-prefill` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--no-enable-cumem-allocator` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--no-enable-dbo` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--no-enable-elastic-ep` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--no-enable-ep-weight-filter` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-enable-eplb` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-enable-expert-parallel` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--no-enable-fault-tolerance` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-enable-flash-late-interaction` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-enable-flashinfer-autotune` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--no-enable-force-include-usage` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-enable-layerwise-nvtx-tracing` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-enable-log-deltas` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--no-enable-log-outputs` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--no-enable-log-requests` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--no-enable-logging-iteration-details` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--no-enable-lora` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--no-enable-mamba-cache-stochastic-rounding` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--no-enable-mfu-metrics` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--no-enable-mixed-moe-lora-format` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--no-enable-mm-embeds` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--no-enable-moe-shared-loras` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--no-enable-offline-docs` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-enable-per-request-metrics` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--no-enable-prefix-caching` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--no-enable-prompt-embeds` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-enable-prompt-tokens-details` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-enable-request-id-headers` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-enable-return-routed-experts` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--no-enable-server-load-tracking` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-enable-sleep-mode` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-enable-ssl-refresh` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--no-enable-tokenizer-info-endpoint` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-enable-tower-connector-lora` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--no-enforce-eager` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--no-exclude-tools-when-tool-choice-none` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-fail-on-environ-validation` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-fully-sharded-loras` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--no-interleave-mm-strings` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--no-jit-monitor-verbose` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-kv-cache-metrics` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--no-kv-sharing-fast-prefill` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--no-language-model-only` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-log-error-stack` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--no-mm-encoder-only` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--no-numa-bind` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-ray-workers-use-nsight` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--no-return-tokens-as-token-ids` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-scheduler-reserve-full-isl` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--no-skip-mm-profiling` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--no-skip-tokenizer-init` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-specialize-active-lora` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--no-tokens-only` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-trust-remote-code` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-trust-request-chat-template` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-use-fp64-gumbel` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-use-replayssm` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--no-use-tqdm-on-load` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--node-rank` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--num-gpu-blocks-override` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--numa-bind` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--numa-bind-cpus` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--numa-bind-nodes` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--offload-backend` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--offload-group-size` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--offload-num-in-group` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--offload-params` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--offload-prefetch-step` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--optimization-level` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--otlp-traces-` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--otlp-traces-endpoint` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--override-attention-dtype` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--override-generation-config` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--performance-mode` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--physcpubind` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--pipeline-parallel-size` | not-applicable | experiment has one GPU |
| `--pooler-config` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--port` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--prefill-context-parallel-size` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--prefill-schedule-interval` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--prefix-caching-hash-algo` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--prefix-match-unit` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--profiler-config` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--pt-load-map-location` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--quantization` | fixed-invariant | NVFP4 ModelOpt weights are immutable |
| `--quantization-config` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--ray-workers-use-nsight` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--reasoning-config` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--reasoning-parser` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--reasoning-parser-plugin` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--renderer-num-workers` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--replayssm-buffer-len` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--response-role` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--return-tokens-as-token-ids` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--revision` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--root-path` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--runner` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--safetensors-load-strategy` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--safetensors-prefetch-block-size` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--safetensors-prefetch-num-threads` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--scheduler-cls` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--scheduler-reserve-full-isl` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--scheduling-policy` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--seed` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--served-model-name` | fixed-invariant | served identity is immutable |
| `--show-hidden-metrics-for-` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--show-hidden-metrics-for-version` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--shutdown-timeout` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--skip-mm-profiling` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--skip-tokenizer-init` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--spec-method` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--spec-model` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--spec-tokens` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--specialize-active-lora` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--speculative-config` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--ssl-ca-certs` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--ssl-cert-reqs` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--ssl-certfile` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--ssl-ciphers` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--ssl-keyfile` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--stream-interval` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--structured-outputs-config` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--tensor-parallel-size` | not-applicable | experiment has one GPU |
| `--tokenizer` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--tokenizer-mode` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--tokenizer-revision` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--tokens-only` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--tool-call-parser` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--tool-parser-plugin` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--tool-server` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--trust-remote-code` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--trust-request-chat-template` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--ubatch-size` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--uds` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--use-fp64-gumbel` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--use-replayssm` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--use-tqdm-on-load` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--uvicorn-log-level` | applicable-control | observability or diagnostics can quantify overhead but cannot define the workload |
| `--video-pruning-method` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--video-pruning-rate` | not-applicable | belongs to a model, topology, modality, or service feature absent from this workload |
| `--watermark` | applicable-search | belongs to an applicable scheduling, cache, kernel, graph, or frontend family |
| `--weight-transfer-config` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--worker-cls` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
| `--worker-extension-cls` | not-applicable | audited against the single-GPU dense text-only launch; this option does not alter the applicable serving hot path or is fixed by the checkpoint/API contract |
