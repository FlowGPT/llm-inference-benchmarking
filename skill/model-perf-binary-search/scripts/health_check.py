#!/usr/bin/env python3
"""Pre-flight health check for model-perf-binary-search.

Runs at session start via bootstrap.sh. Verifies:
  1. GPU visible, driver loaded, nvidia-smi works.
  2. PCIe link is not severely downgraded (critical for native offload).
  3. Pinned host <-> GPU bandwidth matches the link's theoretical capacity.
  4. No serious AER errors on the GPU's PCIe path.
  5. Sufficient free GPU memory and disk space for a benchmark run.

Exit codes:
  0 - All green. Safe for both offload and non-offload runs.
  1 - Warnings only. Non-offload runs OK; offload metrics may be misleading.
  2 - Red. Offload runs strongly discouraged; only proceed with --force.

Outputs (default):
  - Human-readable report to stderr.
  - Machine-readable JSON to the path given by --json-out (default
    <workdir>/.health_check.json so the agent can grep it).

Skip the bandwidth measurement with --no-bandwidth (saves ~2s) when you only
need the structural checks.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# PCIe x16 unidirectional theoretical throughput (GB/s), after encoding.
PCIE_THEORY_GBPS = {1: 4.0, 2: 8.0, 3: 15.75, 4: 31.5, 5: 63.0, 6: 126.0}

ERR = []  # collected issues: list of (severity, message)


def warn(msg):  ERR.append(("warn", msg))
def red(msg):   ERR.append(("red", msg))


def run(cmd, **kw):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, **kw)
    except Exception:
        return None


def get_gpu_info():
    r = run(["nvidia-smi",
             "--query-gpu=name,memory.total,memory.free,driver_version,"
             "pcie.link.gen.current,pcie.link.gen.max,"
             "pcie.link.width.current,pcie.link.width.max",
             "--format=csv,noheader"])
    if not r or r.returncode != 0:
        return None
    # Multi-GPU hosts print one CSV row per device; take the first only.
    first_line = next((ln for ln in r.stdout.splitlines() if ln.strip()), "")
    parts = [x.strip() for x in first_line.split(",")]
    if len(parts) < 8:
        return None
    name, mem_tot, mem_free, drv, cur_g, max_g, cur_w, max_w = parts[:8]
    return {
        "name": name,
        "memory_total_mib": int(mem_tot.split()[0]),
        "memory_free_mib":  int(mem_free.split()[0]),
        "driver": drv,
        "link_gen_current":   int(cur_g),
        "link_gen_max":       int(max_g),
        "link_width_current": int(cur_w),
        "link_width_max":     int(max_w),
    }


def get_gpu_bdf():
    r = run(["bash", "-c",
             "lspci | grep -iE 'VGA|3D' | grep -i nvidia | awk '{print $1}' | head -1"])
    return r.stdout.strip() if r and r.returncode == 0 else None


def get_aer_counts(bdf):
    counts = {"correctable": 0, "fatal": 0, "nonfatal": 0}
    if not bdf:
        return counts
    dev = Path(f"/sys/bus/pci/devices/0000:{bdf}")
    paths = [dev]
    up = dev.resolve().parent if dev.exists() else None
    if up and up.exists() and (up / "vendor").exists():
        paths.append(up)
    for d in paths:
        for kind in ("correctable", "fatal", "nonfatal"):
            f = d / f"aer_dev_{kind}"
            if not f.exists():
                continue
            try:
                for line in f.read_text().splitlines():
                    a = line.split()
                    if len(a) == 2 and a[0].startswith("TOTAL_") and a[1].isdigit():
                        counts[kind] += int(a[1])
            except Exception:
                pass
    return counts


def detect_virt():
    r = run(["lspci"])
    if r and r.returncode == 0 and "Virtio GPU" in r.stdout:
        return "virtio-gpu detected (KVM/QEMU)"
    if Path("/sys/hypervisor/type").exists():
        try:
            return f"hypervisor: {Path('/sys/hypervisor/type').read_text().strip()}"
        except Exception:
            pass
    r = run(["systemd-detect-virt"])
    if r and r.returncode == 0:
        out = r.stdout.strip()
        if out and out != "none":
            return f"systemd-detect-virt: {out}"
    return None


def get_running_gpu_procs():
    r = run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"])
    if not r or r.returncode != 0 or not r.stdout.strip():
        return []
    procs = []
    for line in r.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3:
            try:
                procs.append({"pid": int(parts[0]), "name": parts[1],
                              "memory_mib": int(parts[2].split()[0])})
            except Exception:
                pass
    return procs


def measure_pinned_bw(skip_alloc_mib=0):
    """Returns dict {size_mib: {h2d, d2h}} or None if torch / CUDA unavailable.
    skip_alloc_mib: if non-zero, the GPU already has this much memory in use,
                    so skip sizes that would OOM.
    """
    try:
        import torch
    except ImportError:
        return None, "torch not importable"
    if not torch.cuda.is_available():
        return None, "torch.cuda.is_available() is False"

    dev = torch.device("cuda:0")
    free_mib = torch.cuda.mem_get_info(0)[0] // (1024 * 1024)
    sizes = [m for m in (16, 64) if m * 2 < free_mib]
    if not sizes:
        return None, f"GPU has only {free_mib} MiB free; can't run even a 16 MiB probe"

    results = {}
    for mib in sizes:
        n = mib * 1024 * 1024 // 4
        try:
            host = torch.empty(n, dtype=torch.float32, pin_memory=True)
            gbuf = torch.empty(n, dtype=torch.float32, device=dev)
        except RuntimeError as e:
            results[mib] = {"error": str(e)[:80]}
            continue

        def time_copy(direction, iters=5, warmup=2):
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            for _ in range(warmup):
                (gbuf.copy_ if direction == "h2d" else host.copy_)(
                    host if direction == "h2d" else gbuf, non_blocking=True)
            torch.cuda.synchronize()
            for i in range(iters):
                starts[i].record()
                (gbuf.copy_ if direction == "h2d" else host.copy_)(
                    host if direction == "h2d" else gbuf, non_blocking=True)
                ends[i].record()
            torch.cuda.synchronize()
            ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
            return (mib * 1024 * 1024 / 1e9) / (ms[len(ms) // 2] / 1000.0)

        results[mib] = {"h2d": time_copy("h2d"), "d2h": time_copy("d2h")}
        del host, gbuf
        torch.cuda.empty_cache()
    return results, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-bandwidth", action="store_true",
                    help="skip the pinned H2D/D2H measurement")
    ap.add_argument("--json-out", type=str, default=None,
                    help="write machine-readable result here (default: $LLM_BENCH_DIR/.health_check.json)")
    ap.add_argument("--workdir", type=str, default=os.environ.get("LLM_BENCH_DIR"))
    args = ap.parse_args()

    json_out = args.json_out
    if not json_out and args.workdir and Path(args.workdir).exists():
        json_out = str(Path(args.workdir) / ".health_check.json")

    p = lambda s="": print(s, file=sys.stderr)
    p("=" * 62)
    p(" Health check: model-perf-binary-search")
    p("=" * 62)

    # 1. GPU presence + link state
    gpu = get_gpu_info()
    if gpu is None:
        red("nvidia-smi failed — driver or CUDA not loaded")
        p("✗ no GPU visible to nvidia-smi")
        if json_out:
            Path(json_out).write_text(json.dumps({"exit": 2, "errors": ["nvidia-smi failed"]}))
        return 2

    cur_g, max_g = gpu["link_gen_current"], gpu["link_gen_max"]
    theory_cur = PCIE_THEORY_GBPS.get(cur_g, 0)
    theory_max = PCIE_THEORY_GBPS.get(max_g, 0)

    p(f"GPU       : {gpu['name']}  ({gpu['memory_total_mib']} MiB total, "
      f"{gpu['memory_free_mib']} MiB free), driver {gpu['driver']}")
    p(f"PCIe link : Gen{cur_g} x{gpu['link_width_current']}  "
      f"(hardware max: Gen{max_g} x{gpu['link_width_max']}, "
      f"theoretical: ~{theory_cur:.0f} GB/s now / ~{theory_max:.0f} GB/s max)")

    virt = detect_virt()
    p(f"Platform  : {virt or 'bare metal'}")

    # 2. PCIe link gen evaluation
    if cur_g < max_g:
        ratio = theory_max / theory_cur if theory_cur else 0
        sev = red if cur_g <= 2 else warn
        sev(f"PCIe link downgraded from Gen{max_g} to Gen{cur_g} "
            f"({ratio:.1f}× slower, ~{theory_cur:.0f} GB/s vs ~{theory_max:.0f} GB/s). "
            + ("Hypervisor-side issue (vfio-pci passthrough); fix at host level."
               if virt else
               "Check BIOS PCIe gen setting / riser cables / slot seating."))

    # 3. AER errors
    bdf = get_gpu_bdf()
    aer = get_aer_counts(bdf) if bdf else {"correctable": 0, "fatal": 0, "nonfatal": 0}
    total = sum(aer.values())
    p(f"AER errs  : correctable={aer['correctable']}, fatal={aer['fatal']}, "
      f"nonfatal={aer['nonfatal']}  ({'clean' if total == 0 else 'see below'})")
    if aer["fatal"] > 0 or aer["nonfatal"] > 0:
        red(f"AER non-zero fatal/nonfatal — flaky PCIe hardware, replace card or fix slot")
    elif aer["correctable"] > 1000:
        warn(f"AER correctable={aer['correctable']} is high — link signal integrity marginal")

    # 4. Running GPU processes (warn if something is hogging memory)
    procs = get_running_gpu_procs()
    if procs:
        p(f"GPU procs : {len(procs)} active")
        for pr in procs:
            p(f"            PID {pr['pid']}  {pr['name']}  ({pr['memory_mib']} MiB)")
        big = [pr for pr in procs if pr["memory_mib"] > gpu["memory_total_mib"] * 0.5]
        if big:
            pids = ", ".join(str(pr["pid"]) for pr in big)
            warn(f"another process is holding >50% of GPU memory (PID {pids}); "
                 "kill it before serving the benchmark target, or it will OOM the server")
    else:
        p("GPU procs : none")

    # 5. Pinned bandwidth
    bw_data = None
    if args.no_bandwidth:
        p("Pinned BW : skipped (--no-bandwidth)")
    else:
        bw_data, why = measure_pinned_bw()
        if bw_data is None:
            p(f"Pinned BW : skipped ({why})")
            if "torch" in (why or ""):
                warn("torch not available — pinned bandwidth not measured. "
                     "Install with `pip install --user torch` to enable this check.")
        else:
            p("Pinned BW : H2D / D2H (median of 5 iters)")
            best = 0
            for mib, r in bw_data.items():
                if "error" in r:
                    p(f"            {mib:>3} MiB:  alloc failed ({r['error']})")
                    continue
                util = (r["h2d"] / theory_cur * 100) if theory_cur else 0
                p(f"            {mib:>3} MiB:  {r['h2d']:6.2f} / {r['d2h']:6.2f} GB/s   "
                  f"({util:.0f}% of Gen{cur_g} theoretical)")
                best = max(best, r["h2d"])
            # Check if even the current gen's capacity isn't being used
            if theory_cur and best > 0 and best < 0.55 * theory_cur:
                warn(f"measured peak {best:.1f} GB/s is only {best/theory_cur*100:.0f}% "
                     f"of Gen{cur_g} theoretical {theory_cur:.0f} GB/s — CPU/NUMA bottleneck "
                     "or hypervisor emulation overhead beyond just the link gen cap")
            # Absolute threshold for offload usability
            if best > 0 and best < 15.0:
                red(f"pinned bandwidth {best:.1f} GB/s is below the 15 GB/s offload-usability "
                    "floor — native offload will be the dominant bottleneck, "
                    "any offload tuning on this machine will produce misleading results")

    # 6. Disk space for bench-runs
    if args.workdir:
        try:
            usage = shutil.disk_usage(args.workdir)
            free_gb = usage.free / (1024 ** 3)
            p(f"Disk free : {free_gb:.1f} GiB at {args.workdir}")
            if free_gb < 5:
                warn(f"only {free_gb:.1f} GiB free at workdir; "
                     "bench-runs jsonls + service logs may fill up disk during long sessions")
        except Exception:
            pass

    # Summary
    p("")
    reds  = [m for s, m in ERR if s == "red"]
    warns = [m for s, m in ERR if s == "warn"]
    if reds:
        ec = 2
        p(f"❌ {len(reds)} blocker(s) — offload-unsafe:")
        for m in reds: p(f"   • {m}")
    if warns:
        if not reds: ec = 1
        p(f"⚠  {len(warns)} warning(s):")
        for m in warns: p(f"   • {m}")
    if not reds and not warns:
        ec = 0
        p("✓ all green — ready for both offload and non-offload runs")
    p("")
    p(f"exit: {ec}   "
      f"(0=green, 1=warn-non-offload-ok, 2=blocker-offload-unsafe)")
    if json_out:
        try:
            Path(json_out).write_text(json.dumps({
                "exit": ec,
                "gpu": gpu,
                "virt": virt,
                "aer": aer,
                "gpu_procs": procs,
                "bandwidth": bw_data,
                "issues_red": reds,
                "issues_warn": warns,
            }, indent=2))
            p(f"json result written to: {json_out}")
        except Exception as e:
            p(f"(warn: could not write json output: {e})")

    return ec


if __name__ == "__main__":
    sys.exit(main())
