# Kokkos Hotspot

Kokkos Hotspot is a profiling toolkit to identify the most expensive Kokkos
kernels in an application.

## Installation

Prerequisites:

- CMake >= 3.16
- C++20 compiler
- Python 3
- Kokkos source submodule initialized (`git submodule update --init --recursive`)

Build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Usage

Run capture + analysis:

```bash
./scripts/analyze_hotspots.sh
```

Reuse an existing timing CSV:

```bash
KOKKOS_HOTSPOTS_NO_RUN=1 ./scripts/analyze_hotspots.sh /path/to/timings.csv
```

Main outputs (next to input CSV):

- `kernel_summary.csv`
- `kernel_hotspots.csv`
- `kernel_hotspots.png` (if `matplotlib` is installed)

## Documentation

- Build and configuration: [`docs/build.md`](docs/build.md)
- Capture and analysis workflow: [`docs/usage.md`](docs/usage.md)
- CSV format reference: [`docs/csv-format.md`](docs/csv-format.md)
- Docker usage: [`docs/docker.md`](docs/docker.md)
- Implementation overview: [`docs/roadmap.md`](docs/roadmap.md)
