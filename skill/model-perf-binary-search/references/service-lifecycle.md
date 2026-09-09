# Service Lifecycle and Monitoring

Read this reference before starting, replacing, or monitoring a benchmark
service.

## Ownership

Start only the command supplied or approved by the user. Record the process PID
or an explicit container name and the log path. Never stop unrelated processes
or containers.

Start one service for the complete binary search and reuse it across QPS
probes. Leave it running at the end, including after client failure or user
interrupt. Clean up only replay clients started by this task.

Replacing a service for an approved tuning configuration requires explicit
consent immediately before the stop. Try a graceful stop first; escalate only
for the same owned target.

## Readiness

Poll `GET http://localhost:<port>/v1/models` until it succeeds. Connection
errors and 5xx responses mean not ready. Use a normal budget of about ten
minutes and at least fifteen minutes for CPU KV offload. If readiness fails,
show the last 50 service-log lines and stop.

During load, verify that the owned process or container is still alive. Do not
treat one short HTTP timeout as death: under saturation, confirm liveness with
the process or container state and retry with a longer timeout.

## Docker services

Convert a foreground user command into one owned detached container with a
unique benchmark name. Preserve GPU, IPC, and ulimit arguments required by the
user command. For pinned-memory/offload workloads ensure
`--ulimit memlock=-1` and `--ulimit stack=67108864` are present.

Mount the user's existing model cache only when it is already part of the
approved command or the user approves the added mount. Capture
`docker logs -f <name>` into `bench-runs/service_<timestamp>.log`.

Use the exact container name for liveness and stop operations. Never use broad
container cleanup.

## Health gate

The bootstrap writes `.health_check.json`. Apply its top-level `exit`
exactly as described in `SKILL.md`. When warnings or blockers exist, show the
corresponding issue strings verbatim.

For CPU KV offload:

- pinned-memory permission is mandatory;
- reserve at least `1.4 * offload_size + 15 GiB` available host memory as an
  initial heuristic;
- allow roughly 2.5 seconds per GiB for pinning in the readiness budget;
- monitor available memory and the service's CPU tensor-allocation log lines.

If requested offload cannot fit, ask the user to reduce it or free memory.
Never silently lower the requested value.

## Progress

Record session start time and the next report deadline:

- offload OFF: every 15 minutes;
- offload ON: every 30 minutes.

At each deadline report elapsed time, probes completed, the current bracket and
best passing QPS, the active probe and latest round, an estimated remaining
time, and owned-service liveness. Surface crashes, OOM, readiness failure, and
client storms immediately rather than waiting for the next interval.

## Failure classification

- Cannot start or become ready: deployment failure; stop and report logs.
- Dies under a QPS after a lower QPS passed: that QPS is FAIL; obtain approval
  before relaunching the owned service and continue below it.
- Analyzer reports insufficient rounds: probe FAIL with partial artifacts
  preserved.
- High tail variance: keep the defined primary metric but flag instability and
  recommend a confirmation run.
