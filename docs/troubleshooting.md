# Troubleshooting

## Common Capture Errors

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `error: missing app binary: ...` | `build/kokkos_app` not built or wrong `KOKKOS_APP`. | Build examples (`cmake --build build`) or set `KOKKOS_APP` to a valid executable. |
| `error: missing profiling tool library: ...` | `libkokkos_profiling_tool.so` missing or wrong `KOKKOS_TIMING_TOOL_LIB`. | Build the project and point `KOKKOS_TIMING_TOOL_LIB` to the shared library path. |
| `error: analyze script not found: ...` | Wrong `KOKKOS_TIMING_ANALYZE_SCRIPT`. | Point to `scripts/analyze_timings.py` or a valid analyzer script. |
| `error: timing CSV not found: ...` | Capture did not run or wrote elsewhere. | Check `KOKKOS_TIMING_OUT`, ensure app ran, and verify path permissions. |
| CSV file exists but is empty | App ended before kernel callbacks or output open failed. | Enable `KOKKOS_TIMING_DEBUG=1` and check stderr for tool messages. |

## Common Analysis Errors

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `error: no rows read from CSV files` | CSV has header only or is empty. | Re-run capture with a workload that launches kernels. |
| `error: no valid duration data found` | CSV rows are malformed or duration columns missing/invalid. | Inspect `duration_ns` values and CSV header format. |
| `plot: skipped (matplotlib missing)` | `matplotlib` is not installed in Python environment. | Install matplotlib, then rerun analysis. |

## GPU Timing Specific Issues

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| GPU columns are empty (`gpu_*`) | CPU-only build, disabled events, or runtime/device unavailable. | Use a CUDA/HIP build, set `KOKKOS_TIMING_GPU_EVENTS=1`, and verify GPU runtime access. |
| Debug log: `no GPU runtime/device available` | No visible GPU at runtime or driver/runtime mismatch. | Check device visibility and runtime installation, or run without GPU metrics. |
| Debug logs mention event create/record/synchronize failures | CUDA/HIP event API failed. | Verify runtime health and rerun with `KOKKOS_TIMING_DEBUG=1` for details. |

## Useful Debug Commands

```bash
# Show which files the script expects
echo "APP=$KOKKOS_APP"
echo "TOOL=$KOKKOS_TIMING_TOOL_LIB"
echo "OUT=$KOKKOS_TIMING_OUT"

# Run with debug logs from the profiling tool
KOKKOS_TIMING_DEBUG=1 ./scripts/analyze_hotspots.sh

# Analyze an existing CSV without capture
KOKKOS_HOTSPOTS_NO_RUN=1 ./scripts/analyze_hotspots.sh /path/to/timings.csv
```
