# Replay KV-Eviction Forwarding Design

## Goal

Represent truncation per request in replay datasets while making forwarding to
vLLM an explicit benchmark choice.

## Interface

Each replay request stores a boolean at `body.enable_kv_evict`.

`online_replay.py` adds `--forward-kv-evict`, defaulting to disabled:

- disabled: ignore `body.enable_kv_evict`;
- enabled: copy the boolean to the top-level OpenAI request extension
  `enable_kv_evict`.

`X-Flow-Conversation-Id` remains present in both modes. The replay client must
not synthesize or forward `kv_transfer_params.conversation_id/truncated`,
because the vLLM server derives those internal fields from the trusted header
and top-level flag.

## Compatibility

Restore the tracked branch version of `online_replay.py` before editing. Keep
its production sampling defaults, request timeout, ordered QPS replay, and
per-request sampling behavior. Retain an explicit `--disable-min-p` option for
MTP benchmarks so probing does not add a request or send an unsupported field.

## Verification

- Unit tests cover default omission, exact boolean forwarding, required header,
  and absence of client-generated truncation `kv_transfer_params`.
- The aligned Gemma dataset must contain a boolean field on every request.
- A live disabled/enabled probe must show identical successful request counts;
  enabled mode must produce server truncation-eviction logs only for requests
  whose dataset flag is true.
