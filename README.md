# Kokkos Hotspot

Toolkit to identify the most expensive Kokkos kernels in you application.

This repository contains:

- A Kokkos Tools profiling library (`libkokkos_profiling_tool.so`) that writes
  timings to CSV
- A small sample app (`kokkos_app`) to generate traces
- Analysis scripts to rank hotspots and generate a plot

## Prerequisites

- CMake >= 3.16
- Python 3
- Kokkos with profiling enabled (`Kokkos_ENABLE_PROFILING=ON`)
- To build `libkokkos_profiling_tool.so`, Kokkos must be built with PIC
  (`-DCMAKE_POSITION_INDEPENDENT_CODE=ON`)
- Optional: `matplotlib` to generate the hotspot PNG

## Build

```bash
cmake -S . -B build -DKokkos_DIR=/opt/kokkos/lib/cmake/Kokkos \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Option:

- `-DKOKKOS_BUILD_EXAMPLES=OFF` to build only the profiling library

## Quick Start

Run capture + analysis in one command:

```bash
./scripts/analyze_hotspots.sh
```

This command:

1. runs `build/kokkos_app` with `build/libkokkos_profiling_tool.so`
2. writes a timing CSV
3. generates hotspot analysis outputs

## Manual Timing Capture

```bash
export KOKKOS_TOOLS_LIBS="$PWD/build/libkokkos_profiling_tool.so"
export KOKKOS_TIMING_OUT="$PWD/build/kokkos_kernel_times.csv"
./build/kokkos_app
```

Useful tool environment variables:

- `KOKKOS_TIMING_OUT`: output CSV path
- `KOKKOS_TIMING_APPEND=1`: append to an existing CSV
- `KOKKOS_TIMING_FLUSH=1`: flush after each row
- `KOKKOS_TIMING_GPU_EVENTS=0`: disable CUDA/HIP GPU event timings
- `KOKKOS_TIMING_DEBUG=1`: print debug logs to `stderr`

Useful sample-app environment variables:

- `KOKKOS_ROUNDS`: number of loop iterations (default: `100000`)
- `KOKKOS_APP_SUMMARY`: optional app-level summary CSV path

## Hotspot Analysis

Reuse an existing timing CSV:

```bash
KOKKOS_HOTSPOTS_NO_RUN=1 ./scripts/analyze_hotspots.sh /path/to/timings.csv
```

Useful script environment variables:

- `KOKKOS_APP`: application binary path (default: `build/kokkos_app`)
- `KOKKOS_TIMING_TOOL_LIB`: profiling library path (default: `build/libkokkos_profiling_tool.so`)
- `KOKKOS_TIMING_OUT`: timing CSV path
- `KOKKOS_TIMING_ANALYZE_SCRIPT`: analysis script path (default: `scripts/analyze_timings.py`)
- `KOKKOS_HOTSPOTS_TOP`: number of kernels in the top list

Generated outputs (in the input CSV directory):

- `kernel_summary.csv`: complete per-kernel statistics
- `kernel_hotspots.csv`: top-kernel table
- `kernel_hotspots.png`: visualization (if `matplotlib` is available)

## CSV Format Produced by the Tool

```text
seq,kid,name,type,thread_id,begin_ns,end_ns,duration_ns,dispatch_end_ns,submit_ns,wait_ns,gpu_begin_ns,gpu_end_ns,gpu_duration_ns,scheduling_latency_ns
```

Key fields:

- `name`: kernel label
- `duration_ns`: synchronized CPU duration (`Kokkos::fence()` at event end)
- `gpu_duration_ns`: CUDA/HIP GPU-event duration (when available)
- `scheduling_latency_ns`: `max(duration_ns - gpu_duration_ns, 0)`

## Docker

CPU image:

```bash
docker build -t kokkos-hotspot .
```

CUDA image:

```bash
docker build --network host -f Dockerfile.cuda \
  --build-arg KOKKOS_ARCH=ADA89 \
  -t kokkos-hotspot-cuda .
```
