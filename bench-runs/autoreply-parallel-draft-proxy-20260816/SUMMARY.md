# AutoReply parallel-draft proxy benchmark

Date: 2026-08-16

## Decision

Do not train EAGLE3, DFlash, or DSpark for the fixed AutoReply workload on the
current vLLM 0.27.1 / RTX 5090 deployment. At a deliberately optimistic 60%
aggregate acceptance rate, the best candidate is DSpark BF16 at 7.8 HTTP QPS,
17.9% below the established 9.5-QPS no-speculative baseline. Even DSpark's
unattainable 100%-acceptance cost ceiling fails at 9.0 and 9.5 QPS.

This is a runtime-cost conclusion, not a draft-quality result. Proxy weights
are placeholders; only vLLM's rejection decision is synthetic. The real draft
layers, draft attention/KV, proposer, target verification, scheduler, and CUDA
paths execute.

## Fixed contract and validity

- Dataset: `/mnt/shared/sss/data/auto-reply-test.json`, 1,000 rows, SHA-256
  `501742f94ea6391c74fd624e291bb3cda472d2519a626f8d2a3813a0970eb652`.
- Target: 40-layer, hidden-size 5120 Mistral checkpoint in ModelOpt NVFP4;
  target KV is FP8 on one RTX 5090 (32,607 MiB).
- Request sampling was unchanged: `n=3`, `max_tokens=50`, temperature 0.7,
  top-p 0.8, top-k -1, min-p disabled, frequency/presence penalty 0.01, stop
  `<|im_end|>`.
- `max-model-len=8192` was unchanged.
- Pass means 100% HTTP success and client p50 E2E strictly below 2 seconds in
  a 30-second fixed-QPS window.
- Every independent cold smoke measured exactly 66.1533% prefix-cache hit
  (`133504 / 201810`), matching the expected production 66%-67% range.
- Boundary searches intentionally reuse one warmed container. Their cumulative
  prefix counters can rise to 68%-69% because a later replay sees KV populated
  by the prior replay. Those cumulative values are not used as workload
  alignment evidence.

Each HTTP request creates three sampled sequences, so sequence throughput is
`3 * HTTP QPS`. The 9.5-QPS baseline is 28.5 sequence QPS; the best 60%
candidate's 7.8 HTTP QPS is 23.4 sequence QPS.

## Capacity results

| Method | Draft | Acceptance | Passing QPS | Passing p50 | Adjacent failing QPS | Failing p50 | Delta vs 9.5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EAGLE3 | BF16 | 50% | 6.7 | 1.821 s | 6.8 | 2.290 s | -29.5% |
| EAGLE3 | patched online FP8 | 50% | 7.0 | 1.734 s | 7.1 | 2.122 s | -26.3% |
| EAGLE3 | BF16 | 55% | 6.8 | 1.954 s | 6.9 | 2.198 s | -28.4% |
| EAGLE3 | patched online FP8 | 55% | 7.1 | 1.717 s | 7.2 | 2.311 s | -25.3% |
| EAGLE3 | BF16 | 60% | 6.9 | 1.913 s | 7.0 | 2.240 s | -27.4% |
| EAGLE3 | patched online FP8 | 60% | 7.2 | 1.710 s | 7.3 | 2.095 s | -24.2% |
| DFlash | BF16 | 60% | 7.2 | 1.823 s | 7.3 | 2.214 s | -24.2% |
| DSpark | BF16 | 60% | 7.8 | 1.845 s | 7.9 | 2.011 s | -17.9% |
| DSpark ceiling | BF16 | 100% | 8.5 lower bound | 1.620 s | 9.0 | 2.535 s | at least -10.5% |

At 9.5 QPS, the 100%-acceptance DSpark ceiling has 100% HTTP success but p50
E2E 2.748 seconds, so it still fails the capacity gate.

The aggregate Prometheus acceptance counters validate the synthetic mechanism:
cold 50% smoke measured 49.46% EAGLE3 BF16, 49.35% EAGLE3 FP8, 49.29%
DFlash, and 48.54% DSpark. The 60% boundary services measured 59.96% DFlash
and 59.44% DSpark, both inside the two-point tolerance.

## Draft quantization

EAGLE3 online FP8 raises capacity by exactly 0.3 HTTP QPS at all three tested
acceptance levels:

| Acceptance | BF16 | FP8 | QPS uplift |
| ---: | ---: | ---: | ---: |
| 50% | 6.7 | 7.0 | 4.48% |
| 55% | 6.8 | 7.1 | 4.41% |
| 60% | 6.9 | 7.2 | 4.35% |

Startup model memory falls from 9.06 GiB to 8.75 GiB, a 0.31-GiB reduction.
Logs confirm `CutlassFP8ScaledMMLinearKernel` for the draft. This remains a
patched diagnostic: native vLLM 0.27.1 loses the target's callable
`hf_overrides` while resolving draft quantization. A minimal patch returning an
empty draft override makes EAGLE3 launch; the minimal diff is saved as
`patches/speculative-hf-overrides.patch`.

DFlash online FP8 is structurally unsupported in this release. Its fused
context-KV path directly consumes quantized QKV storage; FP8 pads the input
dimension to 6144 and the custom linear receives incompatible
`8192x5120 @ 6144x5120` operands. DSpark reuses this backbone and has the same
incompatibility. Consequently no honest FP8 QPS uplift can be assigned to
DFlash or DSpark. Quantizing weights offline cannot by itself repair that
runtime path.

## Why 50%-60% acceptance loses

The workload has no conversational history, and history reuse is not needed to
explain the 66%-67% hit rate. `n=3` expands one HTTP prompt into three sampled
sequences. The first branch establishes that prompt's KV blocks and the two
sibling branches reuse them, giving the structural limit `2 / 3 = 66.67%`.
The cold smoke's 66.1533% is close rather than exactly 66.67% because block
alignment and uncached tail tokens are counted at token granularity. Shared
static templates across independent rows can add some reuse, but they are not
the primary source of the stable value.

This also changes the speculation economics: roughly two thirds of the three
branches' prompt prefill is already removed, while responses are only 50
tokens. There is little long-generation tail over which to amortize draft
startup, scheduling, and verification.

`n=3` is a major multiplier. One HTTP request creates three live sequences;
draft generation and target verification operate on all three. At the baseline
boundary the server sustains 28.5 sampled sequences/s. Speculation must save
enough target decode iterations across all three sequences to pay for its own
work, not merely reduce iterations for one HTTP request.

The target is a 40-layer NVFP4 model. The proxy costs are not negligible next
to that aggressively quantized target:

- EAGLE3 is one BF16 layer but runs recursively for K=3 and its first Q/K/V
  projections consume concatenated target/draft state (`2H`). Its checkpoint
  contains about 382.8M BF16 parameters (765.5 MB).
- DFlash performs one parallel proposal but uses five BF16 draft layers plus
  the non-causal context-KV projection; its checkpoint is 2.88 GB.
- DSpark also uses five BF16 layers, a rank-256 Markov head, and a K=7 target
  verification block; its checkpoint is 3.12 GB. K=7 amortizes it better than
  K=3, explaining the measured 7.8 QPS, but not enough to catch 9.5.

EAGLE3's measured capacity rises only 0.2 QPS from 50% to 60% acceptance. A
local linear extrapolation of the faster FP8 series is 0.02 QPS per acceptance
percentage point; closing the 2.3-QPS gap at 60% would require another 115
percentage points, or an impossible 175% aggregate acceptance. This is only a
local diagnostic, not a physical acceptance model, but the independent DSpark
100%-acceptance ceiling confirms the same conclusion without extrapolation.

The bottleneck is therefore mixed target/draft GPU compute and verification
batch expansion, not missing history reuse and not a client-QPS accounting
error. The server's sequence workload is three times HTTP QPS, while all
reported capacity numbers consistently use HTTP QPS.

## Training recommendation

Do not spend a training run on these architectures for this exact dataset and
serving contract. Achieving 50%-60% acceptance is insufficient, draft FP8 buys
only about 4.4% for EAGLE3, and the best architecture cannot beat no-spec even
under 100% synthetic acceptance.

Reconsider training only after a runtime change removes material draft cost:
a much smaller/fused draft, a supported quantized DFlash/DSpark context-KV
path, fewer sampled completions than `n=3`, or substantially longer outputs.
Any reconsideration should first repeat this synthetic ceiling test; training
should begin only if the cost proxy beats 9.5 QPS with margin.

Placeholder predictions make 95%-100% of sampled completions finish by the
50-token length limit and alter content. Therefore these runs cannot estimate
quality, real acceptance achievable after training, or stop-token behavior.
They can validly reject training on runtime cost because synthetic acceptance
is deliberately favorable while all serving work is real.
