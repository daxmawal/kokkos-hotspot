# Hotspot detection flow

## 1) Runtime instrumentation

The profiling library implements Kokkos Tools callbacks from
`Kokkos_Profiling_C_Interface.h`.

For each kernel event (`parallel_for`, `parallel_reduce`, `parallel_scan`):

1. store `begin_ns` at callback begin
2. call `Kokkos::fence()` at callback end to include async backend work
3. store `end_ns` and compute:
   - `duration_ns = end_ns - begin_ns`
4. optionally measure GPU event duration (`gpu_duration_ns`) on CUDA/HIP
5. export one CSV row per event

## 2) CSV output

Current CSV header:

`seq,kid,name,type,thread_id,begin_ns,end_ns,duration_ns,dispatch_end_ns,submit_ns,wait_ns,gpu_begin_ns,gpu_end_ns,gpu_duration_ns,scheduling_latency_ns`

Key fields:

- `name`: kernel label
- `duration_ns`: synchronized CPU wall-time for the event
- `gpu_duration_ns`: GPU event duration when available
- `scheduling_latency_ns`: `max(duration_ns - gpu_duration_ns, 0)`

## 3) Offline hotspot analysis

`scripts/analyze_timings.py`:

- merges one or more timing CSV files
- groups rows by kernel name
- computes `count`, `median`, `mean`, `total` (CPU + GPU/scheduling when available)
- ranks kernels by total CPU time
- writes:
  - `kernel_summary.csv` (all kernels)
  - `kernel_hotspots.csv` (top N)
- plots `kernel_hotspots.png` if `matplotlib` is installed

`scripts/analyze_hotspots.sh` can run the full pipeline:

- run app with profiling tool
- produce timing CSV
- launch analysis script

## 4) Current scope

This repository is intentionally scoped to hotspot detection only
(timing capture + ranking/visualization).
