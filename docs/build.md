# Build and Configuration

## Prerequisites

- CMake >= 3.16
- C++20 compiler
- Python 3 (for analysis scripts)
- Kokkos source submodule initialized (`git submodule update --init --recursive`)

## Default Build (Bundled Kokkos)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

By default, backend auto-detection enables:

- `CUDA` if a CUDA compiler is found
- otherwise `HIP` if `hipcc` is found
- `OpenMP` when available
- `SERIAL` as fallback

## Build Options

- `-DKOKKOS_BUILD_EXAMPLES=OFF`: build only the profiling library
- `-DKOKKOS_AUTO_BACKENDS=OFF`: disable backend auto-detection

Example with manual backend selection:

```bash
cmake -S . -B build \
  -DKOKKOS_AUTO_BACKENDS=OFF \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_SERIAL=OFF
```

## Backend Compatibility

Kokkos Hotspot captures `parallel_for`, `parallel_reduce`, and
`parallel_scan` callbacks and writes one CSV row per completed event.

When `KOKKOS_USE_SUBMODULE=ON` and `KOKKOS_AUTO_BACKENDS=ON`, backend
selection follows `CMakeLists.txt`:

| Backend | Auto-enable rule | Notes |
| --- | --- | --- |
| CUDA | `CMAKE_CUDA_COMPILER` is found | Forces `Kokkos_ENABLE_HIP=OFF`. |
| HIP | CUDA compiler not found and `hipcc` is found | Forces `Kokkos_ENABLE_CUDA=OFF`. |
| OpenMP | `find_package(OpenMP COMPONENTS CXX)` succeeds | Host backend. |
| SERIAL | Always enabled | Fallback host backend. |

Additional behavior:

- `Kokkos_ENABLE_PROFILING` defaults to `ON` when using the bundled submodule.
- With `KOKKOS_USE_SUBMODULE=OFF`, `KOKKOS_AUTO_BACKENDS=ON` is ignored and
  backend choice comes from your external Kokkos installation.
- GPU event metrics (`gpu_*`, `scheduling_latency_ns`) require:
  - a CUDA or HIP build,
  - a working GPU runtime/device at execution time,
  - `KOKKOS_TIMING_GPU_EVENTS` not set to `0`.

## External Kokkos Installation

Use an already installed Kokkos package:

```bash
cmake -S . -B build \
  -DKOKKOS_USE_SUBMODULE=OFF \
  -DKOKKOS_AUTO_BACKENDS=OFF \
  -DKokkos_DIR=/opt/kokkos/lib/cmake/Kokkos \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Reconfiguring a Build Directory

If you switch backend/toolchain in an existing build directory, clear cached
Kokkos backend options:

```bash
cmake -S . -B build -U Kokkos_ENABLE_.* -DCMAKE_BUILD_TYPE=Release
```
