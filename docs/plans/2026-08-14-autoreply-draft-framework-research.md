# AutoReply Draft-Model Framework Research Plan

> **For agents:** Use `executing-plans` inline. Steps use `- [ ]` checkboxes.

**Goal:** Recommend a maintained framework for eventually training a draft
model compatible with the custom AutoReply Mistral target without performing
training or making unmeasured speedup claims.

**Architecture:** Build an evidence matrix from primary documentation, source,
release notes, and minimal read-only compatibility inspection of the target
checkpoint. Separate confirmed support from inference and open questions, then
derive a primary recommendation, fallback, resource estimate, and future smoke
test.

**Reference implementations:** NVIDIA Model Optimizer speculative-decoding
examples, SpecForge documentation/source, and official serving-framework
speculative-decoding compatibility matrices.

---

## File structure

- Create `docs/autoreply-draft-model-framework-research-2026-08-14.md`.
- Create `bench-runs/autoreply-framework-shootout-20260814/research/sources.json`
  containing URLs, retrieval dates, versions/commits, claim mapping, and no
  copied long-form source text.

## Task 1: Freeze target requirements

**Files:** Create the report and source manifest.

**Steps:**

- [ ] Record the checkpoint architecture, dimensions, vocabulary, quantization
  producer/version, tokenizer/chat-template requirements, request sampling,
  and intended vLLM/TensorRT-LLM/SGLang consumers.
- [ ] Distinguish target training checkpoint requirements from the deployed
  NVFP4 target; do not assume NVFP4 is trainable.
- [ ] Add an evidence legend: confirmed, inferred, unsupported, and unknown.

## Task 2: Evaluate maintained training frameworks

**Files:** Modify the report and source manifest.

**Steps:**

- [ ] Research NVIDIA Model Optimizer EAGLE3/Medusa, SpecForge EAGLE3, and at
  least one maintained alternative using official sources only.
- [ ] For each, record custom Mistral extensibility, online/offline training,
  target freezing, hidden-state/logit requirements, draft vocab support,
  distributed training, checkpoint format, licenses, releases, and maintenance.
- [ ] Map export/load compatibility separately for vLLM, TensorRT-LLM, and
  SGLang; an unsupported or roadmap-only path must be labeled accordingly.

## Task 3: Estimate feasibility and recommend

**Files:** Modify the report.

**Steps:**

- [ ] Estimate GPU memory, GPU-hours range, hidden-state storage range, required
  training data, and engineering work with formulas and explicit assumptions.
- [ ] Analyze suitability for 50-token non-greedy `n=3` serving, emphasizing
  that acceptance rate and batch-dependent verifier cost determine speedup.
- [ ] Recommend one primary framework and one fallback, list blockers, and
  provide a bounded future proof-of-concept with success criteria for load,
  acceptance rate, output-distribution correctness, and QPS improvement.
- [ ] Validate every material claim has a primary-source entry and run a link
  check; do not install or execute any training framework.
