# Capture and Analysis Usage

## Quick Start

Run capture + analysis in one command:

```bash
./scripts/analyze_hotspots.sh
```

By default this command:

- runs `build/kokkos_app`
- loads `build/libkokkos_profiling_tool.so`
- writes timings to `build/kokkos_kernel_times.csv`
- generates hotspot outputs next to that CSV

## Reuse an Existing CSV

```bash
KOKKOS_HOTSPOTS_NO_RUN=1 ./scripts/analyze_hotspots.sh /path/to/timings.csv
```

The first positional argument is the CSV path. If omitted, the script uses:

1. `KOKKOS_TIMING_OUT` when set
2. `build/kokkos_kernel_times.csv` otherwise

## Reproducible Scenarios

### CPU-only capture (Serial backend)

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DKOKKOS_AUTO_BACKENDS=OFF \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=OFF \
  -DKokkos_ENABLE_HIP=OFF
cmake --build build-cpu

KOKKOS_APP="$PWD/build-cpu/kokkos_app" \
KOKKOS_TIMING_TOOL_LIB="$PWD/build-cpu/libkokkos_profiling_tool.so" \
KOKKOS_TIMING_OUT="$PWD/build-cpu/kokkos_kernel_times.csv" \
./scripts/analyze_hotspots.sh
```

### GPU capture (CUDA/HIP build, GPU event timings enabled)

```bash
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-gpu

KOKKOS_APP="$PWD/build-gpu/kokkos_app" \
KOKKOS_TIMING_TOOL_LIB="$PWD/build-gpu/libkokkos_profiling_tool.so" \
KOKKOS_TIMING_OUT="$PWD/build-gpu/kokkos_kernel_times.csv" \
KOKKOS_TIMING_GPU_EVENTS=1 \
./scripts/analyze_hotspots.sh
```

### Reanalyze one or more existing CSV files

```bash
KOKKOS_HOTSPOTS_NO_RUN=1 ./scripts/analyze_hotspots.sh /path/to/timings.csv
python3 scripts/analyze_timings.py /path/to/a.csv /path/to/b.csv --top 30
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

### Kokkos runtime + profiling tool

| Variable | Default | Effect |
| --- | --- | --- |
| `KOKKOS_TOOLS_LIBS` | none | Shared library list loaded by Kokkos. Set this to `.../libkokkos_profiling_tool.so` for manual capture runs. |
| `KOKKOS_TIMING_OUT` | `kokkos_kernel_times.csv` | Output CSV path for kernel timing rows. |
| `KOKKOS_TIMING_APPEND` | `0` | When not `0`, append rows to existing CSV. Header is skipped only when appending to an existing file. |
| `KOKKOS_TIMING_FLUSH` | `0` | When not `0`, flush the CSV stream after every row (safer, slower). |
| `KOKKOS_TIMING_GPU_EVENTS` | `1` | When `0`, disables CUDA/HIP event timing collection. Any other non-empty value enables it. |
| `KOKKOS_TIMING_DEBUG` | `0` | When not `0`, prints debug messages from the tool to `stderr`. |

### Example app

| Variable | Default | Effect |
| --- | --- | --- |
| `KOKKOS_ROUNDS` | `100000` | Number of iterations in the sample loop in `examples/app.cpp`. |
| `KOKKOS_APP_SUMMARY` | unset | Optional output path for a tiny app-level summary CSV. |

### `scripts/analyze_hotspots.sh`

| Variable | Default | Effect |
| --- | --- | --- |
| `KOKKOS_APP` | `build/kokkos_app` | Application executable used for capture. |
| `KOKKOS_TIMING_TOOL_LIB` | `build/libkokkos_profiling_tool.so` | Profiling tool library passed via `KOKKOS_TOOLS_LIBS`. |
| `KOKKOS_TIMING_OUT` | `build/kokkos_kernel_times.csv` | CSV path used for capture and as analyzer input when no positional argument is given. |
| `KOKKOS_TIMING_ANALYZE_SCRIPT` | `scripts/analyze_timings.py` | Analyzer script path. |
| `KOKKOS_HOTSPOTS_NO_RUN` | `0` | When `1`, skips capture and only runs analysis. |
| `KOKKOS_HOTSPOTS_TOP` | unset | If set, passed as `--top` to `analyze_timings.py` (otherwise Python default is `20`). |
