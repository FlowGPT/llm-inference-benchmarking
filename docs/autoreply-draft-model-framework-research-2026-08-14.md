# AutoReply Draft-Model Training Framework Research

Date: 2026-08-14

## Recommendation

Use **NVIDIA Model Optimizer EAGLE3** as the primary training path. It is the
only reviewed project that explicitly lists Mistral under EAGLE3 training,
offers online, offline, and streaming hidden-state workflows, supports a
compressed draft vocabulary, and documents export to TensorRT-LLM and SGLang.
Its provenance also matches the deployed checkpoint, which was produced by
ModelOpt. Use **vLLM Speculators** when vLLM-native checkpoint packaging is the
priority, but gate it with a Mistral smoke test because Mistral training and
deployment are still marked in progress in its published support table. Use
**SpecForge EAGLE3** when SGLang-native training and serving are the priority.

This is a feasibility recommendation, not a speedup claim. No draft was
trained and no acceptance rate or speculative QPS was measured.

## Target requirements

| Property | Value | Consequence |
|---|---:|---|
| Architecture | `MistralForCausalLM` | Prefer an architecture-generic Llama/Mistral EAGLE decoder |
| Layers / hidden / intermediate | 40 / 5120 / 14336 | Target is roughly 12-13B parameters; online BF16 training is not a one-5090 job |
| Attention / KV heads | 32 / 8 | GQA metadata must be preserved by hidden-state extraction |
| Vocabulary | 131072 | Full draft LM head is costly; draft-vocabulary compression is valuable |
| Context limit | 131072 | Training need not use the full limit; production prompts are mostly below 4k with a small 4k-8k tail |
| Deployed weights | ModelOpt NVFP4, 8.3 GiB | Excellent for serving, but not evidence that the frozen teacher can train correctly in 4-bit |
| KV cache | FP8 | Serving property; it is not a draft-training checkpoint format |
| Sampling | non-greedy, `n=3`, 50 tokens | Exact rejection sampling and batch-dependent verifier cost must be tested |

The safest assumption is that training or teacher-hidden extraction requires
the original BF16/FP16 target checkpoint. The deployed NVFP4 checkpoint can be
tested as an inference teacher later, but should not be the only available
source checkpoint for a production draft-training project.

## Framework matrix

| Framework | Custom Mistral training | Training modes | Deployment path | Assessment |
|---|---|---|---|---|
| NVIDIA Model Optimizer | Confirmed in its EAGLE3 support matrix | Online HF Trainer; offline hidden states via HF or TensorRT-LLM; streaming from vLLM; distributed/context parallel | Documented TensorRT-LLM and SGLang modules | Primary recommendation |
| vLLM Speculators | Architecture-generic EAGLE3, but its table still marks Mistral training/deployment in progress | Online/offline hidden states via vLLM; single GPU, DDP and FSDP; reduced draft vocabulary | Native Hugging Face-compatible checkpoint loaded directly by vLLM | Best vLLM-native fallback; custom Mistral gate required |
| SpecForge | Plausible but this exact custom Mistral checkpoint is unvalidated | Online/offline, tensor parallel, FSDP | Native SGLang integration | Strong SGLang-first fallback |
| SafeAILab EAGLE | Possible with custom model integration | Reference EAGLE/EAGLE3 scripts and DeepSpeed | Multiple runtimes implement EAGLE, but custom models require adapting model code/KV-cache handling | Reference implementation, higher engineering cost |
| Medusa/MTP variants | Medusa is supported by ModelOpt; native MTP is not present in this Mistral checkpoint | Add/train heads or architecture-coupled MTP modules | Runtime support differs by framework | Not preferred over EAGLE3 for this target |

Primary evidence: [ModelOpt speculative training example](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/speculative_decoding/README.md),
[vLLM Speculators documentation](https://docs.vllm.ai/projects/speculators/en/stable/),
[SpecForge](https://github.com/sgl-project/SpecForge), and the
[official EAGLE implementation](https://github.com/SafeAILab/EAGLE).

## Why Model Optimizer fits best

1. Its published support matrix explicitly marks Mistral compatible with
   Medusa, EAGLE1/2, and EAGLE3.
2. The target can remain frozen while only the lightweight EAGLE module is
   optimized. ModelOpt exposes hidden-state and logit-distillation controls.
3. Offline mode decouples the large teacher from draft optimization. Streaming
   mode avoids a multi-terabyte hidden-state dump when sufficient serving and
   training GPUs/networking are available.
4. A 131072-token output vocabulary is unusually expensive for a small draft.
   ModelOpt's frequency-calibrated draft-vocabulary mapping directly addresses
   this target.
5. The deployment target includes TensorRT-LLM, so the NVIDIA-maintained
   training/export path has the lowest format-integration risk.

## Resource envelope

These are planning estimates, not measured requirements.

### Online training

A rough dense-parameter calculation places the target around 12-13B
parameters. BF16 weights alone are about 25-26 GiB. Activations, the draft,
gradients, optimizer state, and framework buffers make online colocation
unsafe on the 32-GiB RTX 5090 even with the target frozen. Plan for at least
two 48-80-GiB training GPUs, with more headroom preferred for useful sequence
lengths and batches.

### Offline hidden states

EAGLE3 commonly captures low, middle, and high layer features. For three BF16
hidden states, uncompressed storage is approximately:

```text
tokens × 3 layers × 5120 values × 2 bytes
      = tokens × 30,720 bytes
```

That is about 3.1 TiB per 100 million tokens before metadata, alignment,
temporary shards, or logits. One billion tokens approaches 30.7 TiB. This
agrees with ModelOpt's warning that offline data can require several to tens
of terabytes. Selective token storage, fewer examples, lower storage precision,
or streaming should be evaluated before committing capacity.

### Training duration

The official EAGLE project reports roughly one to two days on eight RTX 3090s
for its reference scale, but this does not transfer directly to this target or
data. Budget an initial 8-GPU, 24-48 hour engineering experiment only after a
small end-to-end smoke run validates extraction, loss, export, and loading.

## Serving compatibility and risks

- TensorRT-LLM documents EAGLE3 linear and dynamic-tree execution. For this
  non-greedy workload, exact rejection-sampling behavior must be enabled and
  validated; an argmax-only shortcut is not equivalent.
- SGLang has the clearest SpecForge-to-runtime path.
- ModelOpt documents deployable speculation modules for TensorRT-LLM and
  SGLang. For vLLM, Speculators offers a direct training and serving path, and
  can also convert external EAGLE3 checkpoints. The exact custom-Mistral load
  path in vLLM 0.27.1 must still be smoke-tested rather than assumed.
- `n=3` triples the number of generated branches per HTTP request. Draft
  overhead and verifier batches can erase benefits at high concurrency even
  when single-request latency improves.
- Fifty output tokens is short. Startup/warmup is irrelevant to steady state,
  but per-request draft setup and low acceptance can dominate the limited
  decode span.
- The custom tokenizer, chat template, and production domain must be used to
  synthesize or regenerate training responses. Generic chat data alone can
  depress acceptance rate on AutoReply traffic.

## Minimal future proof of concept

1. Obtain the exact pre-quantization BF16/FP16 target checkpoint and hash it
   against the deployed model lineage.
2. Use ModelOpt EAGLE3 with the target frozen and 5k-20k representative
   AutoReply conversations. Start with a 32k draft vocabulary calibrated on
   production tokens.
3. Run a tiny online or HF hidden-state extraction job and train only long
   enough to prove decreasing validation loss.
4. Export one draft and load it in TensorRT-LLM. Convert or train a separate
   Speculators-format checkpoint for vLLM; treat vLLM and SGLang loading as
   separate compatibility gates.
5. Require greedy token equality, statistically consistent non-greedy output,
   no `n=3` request-shape change, and an average accepted-token count that is
   materially greater than one.
6. Benchmark QPS with and without EAGLE3 using this exact replay. Continue only
   if the speculative configuration raises the SLO-constrained boundary, not
   merely single-request tokens/second.

## Evidence status

- **Confirmed:** ModelOpt lists Mistral EAGLE3 support and documents
  online/offline/streaming paths, vocabulary compression, and deployable
  TensorRT-LLM/SGLang modules.
- **Confirmed:** vLLM Speculators documents online/offline EAGLE3 training,
  reduced-vocabulary training, FSDP, external-checkpoint conversion, and direct
  vLLM serving; its support table marks Mistral as in progress.
- **Confirmed:** SpecForge is maintained and SGLang-oriented, with
  online/offline, tensor-parallel, and FSDP training advertised by the project.
- **Inferred:** This custom Mistral configuration should fit ModelOpt's generic
  Mistral path because it uses standard dense Mistral fields and no sliding
  window. A smoke test is still required.
- **Unknown:** Whether the 0.45 development-version NVFP4 checkpoint can serve
  as the teacher for stable, high-quality EAGLE3 training without its BF16
  source.

## QPS acceptance criterion for any future draft

The current non-speculative production winner is **9.4 external QPS**, with
9.5 QPS failing the strict p50 E2E SLO. A draft is beneficial only if a fresh
cold-start binary search moves the passing boundary above 9.4 QPS while keeping
the exact request body, model identity, `n=3`, and `max-model-len=8192`.
Report both `new_best_qps - 9.4` and `(new_best_qps / 9.4 - 1) * 100%`.
Acceptance rate or single-request latency alone is not a QPS gain.
