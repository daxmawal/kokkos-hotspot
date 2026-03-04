# Implementation Overview and Roadmap

## Current Architecture

### Capture layer (`src/tool_profiling.cpp`)

- Registers Kokkos profiling callbacks:
  - `kokkosp_begin_parallel_for`, `kokkosp_end_parallel_for`
  - `kokkosp_begin_parallel_reduce`, `kokkosp_end_parallel_reduce`
  - `kokkosp_begin_parallel_scan`, `kokkosp_end_parallel_scan`
- Creates one active record per kernel id (`kid`) at begin callback.
- Finalizes each record at end callback and writes one CSV row.
- Uses `Kokkos::fence("kokkos_profiling")` in end callbacks so
  `duration_ns` is synchronized end-to-end host-observed time.
- Optionally collects CUDA/HIP event timings (`gpu_*`) and derives
  `scheduling_latency_ns`.

### Shared utilities (`src/tool_common.cpp`)

- Environment parsing (`env_flag`, `get_env`)
- CSV escaping
- Debug logging helpers

### Analysis layer (`scripts/analyze_timings.py`)

- Reads one or more timing CSV files.
- Aggregates per-kernel CPU, GPU, and scheduling metrics.
- Produces:
  - `kernel_summary.csv`
  - `kernel_hotspots.csv`
  - `kernel_hotspots.png` when matplotlib is available

### Orchestration (`scripts/analyze_hotspots.sh`)

- Optionally runs capture with app + profiling tool.
- Invokes Python analyzer.
- Supports analyze-only mode (`KOKKOS_HOTSPOTS_NO_RUN=1`).

## Current Scope and Limits

- Kernel kinds covered: `for`, `reduce`, `scan`.
- Output format: appendable CSV with one row per kernel event.
- GPU event timings are available only for CUDA/HIP builds with a working
  runtime/device and `KOKKOS_TIMING_GPU_EVENTS` enabled.
- Thread id is stored as a hash of `std::thread::id`.

## Roadmap

1. Add optional capture filters (kernel-name regex, minimum duration).
2. Add richer aggregation dimensions (by `type`, by thread hash, by phase).
3. Add regression-friendly outputs (JSON summary, deterministic ordering).
4. Add automated validation tests for CSV schema and analyzer compatibility.
5. Add user-facing examples for integrating the tool with external apps.
