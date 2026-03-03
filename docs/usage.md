# Capture and Analysis Usage

## Quick Start

Run capture + analysis in one command:

```bash
./scripts/analyze_hotspots.sh
```

This runs `build/kokkos_app` with `build/libkokkos_profiling_tool.so`,
records timings, then generates hotspot outputs.

## Reuse an Existing CSV

```bash
KOKKOS_HOTSPOTS_NO_RUN=1 ./scripts/analyze_hotspots.sh /path/to/timings.csv
```

## Manual Timing Capture

```bash
export KOKKOS_TOOLS_LIBS="$PWD/build/libkokkos_profiling_tool.so"
export KOKKOS_TIMING_OUT="$PWD/build/kokkos_kernel_times.csv"
./build/kokkos_app
```

## Output Files

Outputs are written next to the input timing CSV:

- `kernel_summary.csv`: full per-kernel statistics
- `kernel_hotspots.csv`: top kernels
- `kernel_hotspots.png`: plot (when `matplotlib` is available)

## Environment Variables

Tool variables:

- `KOKKOS_TIMING_OUT`: output CSV path
- `KOKKOS_TIMING_APPEND=1`: append to existing CSV
- `KOKKOS_TIMING_FLUSH=1`: flush after each row
- `KOKKOS_TIMING_GPU_EVENTS=0`: disable CUDA/HIP GPU event timings
- `KOKKOS_TIMING_DEBUG=1`: debug logs to `stderr`

Sample app variables:

- `KOKKOS_ROUNDS`: loop iterations (default `100000`)
- `KOKKOS_APP_SUMMARY`: optional app summary CSV path

Script variables:

- `KOKKOS_APP`: app path (default `build/kokkos_app`)
- `KOKKOS_TIMING_TOOL_LIB`: tool library path (default
  `build/libkokkos_profiling_tool.so`)
- `KOKKOS_TIMING_OUT`: timing CSV path
- `KOKKOS_TIMING_ANALYZE_SCRIPT`: analyzer path (default
  `scripts/analyze_timings.py`)
- `KOKKOS_HOTSPOTS_TOP`: top-kernel count
