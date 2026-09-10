# Optional Serving Tuning

Read this reference only after the user opts into tuning. Always complete the
user's original startup command as the baseline before testing changes.

## Generic parameter tuning

1. Identify the framework from the entrypoint: vLLM, SGLang, TensorRT-LLM,
   LMDeploy, or another user-confirmed framework.
2. Inspect GPU model, memory, count, compute capability, and interconnect
   topology.
3. Research every supplied flag in official documentation for the installed
   framework version. Mark unknown or undocumented flags instead of guessing.
4. Present a table with current value, proposed value, and a one-line
   hardware/workload justification.
5. Print the complete proposed startup command and obtain approval.
6. After the baseline, request consent to stop the owned service and start the
   approved tuned command.
7. Run a complete binary search with the same dataset, SLO, precision, round
   policy, and reporting contract.

Useful areas to consider only when supported by the framework include batching,
sequence limits, model length, GPU-memory utilization, KV-cache dtype,
quantization, execution graphs, parallelism, chunked prefill, prefix caching,
speculative decoding, logging, and timeout behavior.

## Feature-enablement tuning

When the user asks to enable a named feature they do not already understand:

1. Read the feature's official documentation, design/RFC material, source or
   introducing change, release notes, and official benchmark material.
2. Report source links with a one-line takeaway.
3. Inventory every feature knob with default, valid range, and per-GPU versus
   global scope.
4. Label each existing user flag as independent or interacting, with the
   interaction explained.
5. Propose three to five configurations and estimate total wall time:

| Config | Difference from baseline | Hypothesis | Risk |
|--------|--------------------------|------------|------|
| cfg_0 | Original command, feature off | Baseline | None |
| cfg_1 | Feature on, framework defaults | Isolate feature effect | Low |
| cfg_2 | Tune feature-only knobs | Test feature potential | Medium |
| cfg_3 | Tune feature and interacting user knobs | Joint optimum | Medium |

Obtain approval for the matrix before starting. Every configuration gets its
own full binary search and each service replacement requires fresh consent.
Propose at most one evidence-driven refinement and cap the total near six
configurations unless the user explicitly extends it.

## Comparison

Report:

| Config | Feature parameters | Other changes | Max QPS | Delta | Primary p50 | Notes |
|--------|--------------------|---------------|---------|-------|--------------|-------|

Recommend the strongest tested configuration that meets the same SLO. It is
valid to conclude that the feature does not help this workload.
