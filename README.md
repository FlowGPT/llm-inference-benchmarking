# LLM Inference Benchmarking for Chat 

## Automated SLO capacity search with Codex

This repository includes a repo-local Codex skill at
[`skill/model-perf-binary-search/SKILL.md`](skill/model-perf-binary-search/SKILL.md).
It finds the highest tested QPS at which an OpenAI-compatible LLM service keeps
client-side p50 end-to-end latency strictly below a target SLO. The workflow
probes the initial bounds, expands them when necessary, and then performs a
binary search to the requested precision.

### Prerequisites

- Open this checked-out repository as the Codex workspace.
- Install [`uv`](https://docs.astral.sh/uv/). The skill uses it to create or
  reuse `.venv` and install the benchmark dependencies.
- Put the replay dataset under `/mnt/shared/sss/data`. The skill uses the
  selected file in place and does not copy it into the repository.
- Prepare the full command that starts the model as an OpenAI-compatible
  service. The command must include its listening port.

The checked-out `online_replay.py` must support
`--serialize-conversations`, `--continuous-qps-window`, and
`--preselected-route`. The bootstrap checks these capabilities before it
installs dependencies or starts a benchmark.

### Run the skill

In Codex chat, ask Codex to use the bundled skill rather than a globally
installed copy:

```text
Use the repository skill at skill/model-perf-binary-search/SKILL.md to find
the maximum sustainable QPS for my model under a p50 E2E latency SLO.
```

The skill first lists regular files in `/mnt/shared/sss/data` and asks you to
select exactly one dataset. It then collects the remaining required inputs:

- the complete service startup command;
- initial QPS bounds (`LOW HIGH`);
- offload mode (`ON` or `OFF`);
- the model name passed to the API and the API port;
- tuning mode (`none`, generic parameter tuning, or feature-enablement tuning);
- optional SLO and search precision overrides (defaults: `6.5` seconds and
  `0.1` QPS).

You can provide those values in the next message, for example:

```text
Dataset: /mnt/shared/sss/data/my-replay.jsonl
Service command: docker run ... --port 8000 ...
QPS bounds: 0.5 8.0
Offload: OFF
Model: my-model
Port: 8000
Tuning: none
SLO: 6.5 seconds
Precision: 0.1 QPS
```

During the run, Codex validates the host, starts or reuses the service with the
documented ownership checks, measures prefix-cache activity when exposed, and
classifies every probe from client E2E latency. Non-offload runs use eight
30-second rounds; offload runs use sixteen. Results are written under
`bench-runs/`, and the final report includes every tested QPS, the largest
passing QPS, artifact paths, and the service PID or container. The service is
left running after the search.

### Preflight and self-test

To validate repository compatibility and dataset selection without installing
dependencies or running GPU checks:

```bash
export LLM_BENCH_DATASET_SRC=/mnt/shared/sss/data/my-replay.jsonl
bash skill/model-perf-binary-search/scripts/bootstrap.sh --check-only
```

To run the bundled helper and repository-safety tests:

```bash
bash skill/model-perf-binary-search/scripts/smoke.sh
```

Generated environments, health reports, and benchmark artifacts are ignored
by Git. The skill does not clone repositories, switch branches, or modify Git
state.

### Set up standalone vllm server 

```
docker run -p 8080:8080 --gpus all vllm/vllm-openai --model Nitral-AI/Captain-Eris_Violet-V0.420-12B --max-model-len 10000 --swap-space 4 --dtype auto --enable-chunked-prefill --disable-log-requests --enable-prefix-caching --port 8080 --root-path /api --served-model-name Nitral-AI/Captain-Eris_Violet-V0.420-12B --max-num-seqs 72 --quantization fp8 --max-num-batched-tokens 1024 --kv-cache-dtype fp8
```

----------------------

### Set up synthetic benchmark 

Gives you the option to use a source prompt at the start, and add some random text after it to control the prompt cache percentage. 

```
git clone https://github.com/FlowGPT/llm-inference-benchmarking

cd llm-inference-bench-char

pip install -r requirements.txt

python run.py --rounds 1 -q 0.5 --api-base http://localhost:8080/api/v1 --model Nitral-AI/Captain-Eris_Violet-V0.420-12B  --max-tokens=250 --prompt-file prompt-1k.txt --random-tokens 3000 --use-chat
```
This benchmark runs 0.5 rps on a 13b model with an input of 4.5k tokens and an output of 250 tokens, prefix cache rate 20%. 

source: https://github.com/leptonai/leptonai/blob/main/misc/benchmark/run.py

-----------------------

### Online replay

You can use the online data to replay the production environments.

The online_replay.py script allows you to replay requests from log file with two different modes:

1. Timestamp-based replay (maintains original request timing):
```bash
python online_replay.py --input replay-logs-origin.log --replay-mode timestamp --sample-range 0.0 0.1 \
    --api-base http://localhost:8080/api/v1 --model Nitral-AI/Captain-Eris_Violet-V0.420-12B --round-duration 60
```

2. QPS-based replay (controls request rate):
```bash
python online_replay.py --input replay-logs-origin.log --replay-mode qps --target-qps 5 --sample-range 0.0 0.1 \
    --api-base http://localhost:8080/api/v1 --model Nitral-AI/Captain-Eris_Violet-V0.420-12B --round-duration 60
```

- How to choose between these two modes?

    In multi-instance scenarios, the timestamp mode is recommended. In single-instance scenarios, the qps mode is sufficient. This is because in production environments, requests are routed among multiple instances, making the request arrival pattern appear uniform for each individual instance.

- Despite similar final QPS, why does timestamp mode show higher latency?

    This is caused by non-uniform request arrival patterns. According to queueing theory, when requests arrive non-uniformly (e.g., Poisson process with high variance), bursty requests can lead to temporary queue buildup, significantly increasing the average queuing time.

- What is the detailed logging feature?

    The `--detailed-logs` parameter enables real-time tracking of each request's performance metrics. Each request is assigned a unique ID, and detailed information including send time, TTFT, completion time, token counts, and processing times are recorded in a CSV file. This data is written in real-time to a `log/detailed_results_[timestamp].csv` file, allowing for detailed analysis of request performance patterns.


Key parameters:
- `--input`: Input log file path
- `--replay-mode`: Replay mode (timestamp/qps)
- `--sample-range`: Sampling range [START, END) to control the percentage of requests to send (e.g., 0.0 0.2)
- `--round-duration`: Performance statistics collection period (seconds)
- `--max-rounds`: Maximum number of rounds to run
- `--api-base`: API service endpoint
- `--model`: Model name
- `--max-token`: Maximum token output of the model
- `--use-chat`: Whether to use chat interface
- `--json-output`: Output performance metrics in JSON format
- `--verbose`: Enable detailed logging output (default: False, only show statistics)
- `--detailed-logs`: Enable detailed per-request logging dir path with unique IDs (saved to CSV file)
- `--e2e-slo`: End-to-end latency SLO target in seconds (float). Example: `--e2e-slo 5.0`. When set, the report includes "E2E SLO Attainment" which is the percentage of total requests that are successful and have latency ≤ SLO.
- `--ttft-slo`: TTFT SLO in milliseconds (int). When set, the report includes "TTFT SLO Attainment" which is the percentage of total requests that are successful and have TTFT ≤ threshold.
- `--tpot-slo`: TPOT SLO in milliseconds (int). When set, the report includes "TPOT SLO Attainment" which is the percentage of requests whose time per output token (ms/token) ≤ threshold.


Performance metrics:
- Latency statistics
- Throughput
- TTFT (Time To First Token)
- TPOT (Time Per Output Token)
- Input/Output Tokens per Minute
- Success Rate
- E2E SLO Attainment (if `--e2e-slo` is provided)
- TTFT SLO Attainment (if `--ttft-slo` is provided)
- TPOT SLO Attainment (if `--tpot-slo` is provided)

Notes on display:
- When any SLO thresholds are set, a second line is appended to the table title to summarize the SLO attainment values (E2E/TTFT/TPOT) for quick inspection.

#### Attention
If you want to start a process using online_replay.py to replay qps>10, you'd better split it to multiple terminals and run them separately. By modifying the `--sample-range` parameter, you can specify different sampling ranges for each process. This approach helps avoid client-side issues caused by high concurrency. You can refer to `run_client_split.sh` for implementation details.

For example, to achieve a total QPS of 20, you can:
1. Run the first process with `--target-qps 10  --sample-range 0.0 0.5` in one terminal
2. Run the second process with `--target-qps 10  --sample-range 0.5 1.0` in another terminal

This distributed approach ensures better stability and more accurate benchmarking results.

To stop processes, you can also open a new terminal and run `pkill -f "online_replay.py"`

### Detailed Log Format

When using the `--detailed-logs` parameter, the script generates a CSV file with the following columns:

- `request_id`: Unique identifier for each request (UUID)
- `conversation_id`: Original conversation ID from the log file
- `send_time`: Timestamp when the request was sent
- `ttft_time`: Timestamp when the first token was received
- `total_time`: Timestamp when the response was completed
- `tokens_in`: Number of input tokens
- `tokens_out`: Number of output tokens
- `ttft`: Time to first token (seconds)
- `tpot`: Time per output token (seconds)

JSON output extra fields when SLO flags are provided:
- `ttft_slo_ms`, `ttft_slo_attainment`
- `tpot_slo_ms`, `tpot_slo_attainment`

This data can be analyzed using any CSV-compatible tool such as pandas, Excel, or data visualization software to identify performance patterns, bottlenecks, or unusual behavior in your LLM serving system.
