# TensorRT-LLM Disabled Top-K Adapter Design

Date: 2026-08-15

## Goal

Allow the fixed AutoReply sampling policy (“top-k disabled”) to reach
TensorRT-LLM without changing its semantics. vLLM continues to receive
`top_k=-1`; TensorRT-LLM receives no `top_k` field and therefore uses its
documented OpenAI-server default `top_k=0`, which also disables top-k.

## Evidence and rejected representations

TensorRT-LLM 1.2.1 and 1.3.0rc22 declare the OpenAI chat request field as
`top_k: int = 0`. Their internal `SamplingParams` accepts `None`, but a literal
JSON `null` reaches the OpenAI schema before that internal class and is not a
valid integer. The installed OpenAI Python SDK serializes
`extra_body={"top_k": None}` as `"top_k": null`; therefore merely assigning
Python `None` is insufficient.

Rejected alternatives:

1. Send JSON `null`: expected schema rejection.
2. Send TensorRT-LLM's `top_k=0` explicitly: works but changes the canonical
   request value instead of representing the requested `None` path.
3. Patch the TensorRT-LLM server or add an HTTP proxy: changes the framework or
   adds latency, weakening the fairness comparison.

## Selected design

Add an opt-in replay flag that removes keys whose values are `None` from the
static extra request body immediately before HTTP serialization. The canonical
TensorRT replay command supplies `top_k: null` in the controller configuration
and enables this flag; the actual wire body omits `top_k`. Without the flag,
existing replay behavior remains byte-for-byte unchanged, including the
ability to send explicit JSON null for other workloads.

The TensorRT controller must record both:

- canonical semantic policy: `top_k_disabled=true`;
- wire representation: `top_k` omitted, TensorRT default `0`.

No other request field may be removed or translated.

## Compatibility gate

Before performance testing, require all of the following on the stable image:

1. Request succeeds through `/v1/chat/completions`.
2. Non-streaming response contains choice indices `0,1,2`.
3. Streaming response contains all three indices.
4. Captured wire body contains the exact fixed sampling fields except `top_k`,
   and contains no literal `top_k: null`.
5. Server-side effective sampling evidence shows top-k disabled/default zero.
6. `--max_seq_len 8192`, model, tokenizer, and all other request fields remain
   fixed.

Only after this gate passes may TensorRT-LLM enter launch-parameter screening
and SLO-constrained QPS search.

## Skill update

After the live gate succeeds, add an opt-in TensorRT-LLM note to
`model-perf-binary-search`: when the user's semantic policy is top-k disabled
but their canonical client uses vLLM's `-1` sentinel, offer a TensorRT-specific
wire mapping that omits the field. Never apply it to standard or vLLM sessions,
and never silently translate an enabled top-k value.

## Verification

- Unit test proves the new flag omits only `None` static-extra keys.
- Unit test proves default behavior still forwards explicit `None`.
- Controller test proves the TensorRT replay command enables the adapter and
  preserves every other immutable sampling value.
- Live stable-image compatibility test proves HTTP success and `n=3`.
- Full relevant test suite, JSON validation, and a clean-commit worktree test
  run before commit/push.
