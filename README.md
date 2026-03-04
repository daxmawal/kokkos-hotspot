# Kokkos Hotspot

Kokkos Hotspot is a profiling toolkit to identify the most expensive Kokkos
kernels in an application.

## Quick Start (2 minutes)

Prerequisites:

- CMake >= 3.16
- C++20 compiler
- Python 3
- Kokkos source submodule initialized

Build + capture + analysis:

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./scripts/analyze_hotspots.sh
```

The script runs `build/kokkos_app` with
`build/libkokkos_profiling_tool.so` and writes, next to the timing CSV:

- `kernel_summary.csv`
- `kernel_hotspots.csv`
- `kernel_hotspots.png` (if `matplotlib` is installed)

Analyze an existing timing CSV without rerunning the app:

```bash
KOKKOS_HOTSPOTS_NO_RUN=1 ./scripts/analyze_hotspots.sh /path/to/timings.csv
```

## Use with Your Own Executable

Use `analyze_hotspots.sh` with another executable (no special arguments):

```bash
KOKKOS_APP=/path/to/my_app \
KOKKOS_TIMING_TOOL_LIB=$PWD/build/libkokkos_profiling_tool.so \
KOKKOS_TIMING_OUT=$PWD/build/my_app_times.csv \
./scripts/analyze_hotspots.sh
```

If your executable needs arguments, run capture manually then analyze:

```bash
env KOKKOS_TOOLS_LIBS="$PWD/build/libkokkos_profiling_tool.so" \
    KOKKOS_TIMING_OUT="$PWD/build/my_app_times.csv" \
    /path/to/my_app --my --args

python3 scripts/analyze_timings.py "$PWD/build/my_app_times.csv" --top 30
```

## Documentation

- Build and configuration: [`docs/build.md`](docs/build.md)
- Capture and analysis workflow: [`docs/usage.md`](docs/usage.md)
- CSV format reference: [`docs/csv-format.md`](docs/csv-format.md)
- Troubleshooting: [`docs/troubleshooting.md`](docs/troubleshooting.md)
- Docker usage: [`docs/docker.md`](docs/docker.md)
- Implementation overview: [`docs/roadmap.md`](docs/roadmap.md)
