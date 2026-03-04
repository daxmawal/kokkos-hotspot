# CSV Format Reference

The profiling tool writes one row per kernel event with this header:

```text
seq,kid,name,type,thread_id,begin_ns,end_ns,duration_ns,dispatch_end_ns,submit_ns,wait_ns,gpu_begin_ns,gpu_end_ns,gpu_duration_ns,scheduling_latency_ns
```

All `*_ns` fields are in nanoseconds.

## Column Reference

| Column | Meaning | Formula / notes |
| --- | --- | --- |
| `seq` | Row sequence number | Starts at `1`, increases by one for each completed kernel event. |
| `kid` | Kernel record id | Unique id assigned at begin callback time. |
| `name` | Kernel label | `parallel_for/reduce/scan` name from Kokkos callbacks. May be empty. |
| `type` | Kernel kind | One of `for`, `reduce`, `scan`. |
| `thread_id` | Host thread identity | Hash of `std::thread::id` for the callback thread. |
| `begin_ns` | Kernel begin timestamp | `begin_time - start_time` (tool start clock). |
| `end_ns` | End timestamp after synchronization | Timestamp taken after `Kokkos::fence("kokkos_profiling")` in end callback. |
| `duration_ns` | Total synchronized CPU-side duration | `max(end_ns - begin_ns, 0)`. |
| `dispatch_end_ns` | Timestamp before synchronization | Timestamp taken in end callback just before `Kokkos::fence`. |
| `submit_ns` | Submission phase duration | `max(dispatch_end_ns - begin_ns, 0)`. |
| `wait_ns` | Synchronization/wait phase duration | `max(end_ns - dispatch_end_ns, 0)`. |
| `gpu_begin_ns` | GPU begin timestamp | Device event offset from tool GPU origin event. Empty if unavailable. |
| `gpu_end_ns` | GPU end timestamp | Device event offset from tool GPU origin event. Empty if unavailable. |
| `gpu_duration_ns` | GPU event duration | Device event elapsed time between begin/end events. Empty if unavailable. |
| `scheduling_latency_ns` | CPU minus device time | `max(duration_ns - gpu_duration_ns, 0)` when `gpu_duration_ns` exists; empty otherwise. |

## Missing Values

The following fields can be empty:

- `gpu_begin_ns`
- `gpu_end_ns`
- `gpu_duration_ns`
- `scheduling_latency_ns`

This happens when GPU event timings are unavailable or disabled, for example:

- CPU-only build/runtime
- `KOKKOS_TIMING_GPU_EVENTS=0`
- GPU runtime/device not ready
- GPU event API failure during capture

## Metric Interpretation

- `duration_ns` is end-to-end host-observed duration for a kernel event because the end callback fences before timing `end_ns`.
- `gpu_duration_ns` estimates pure device execution duration between recorded GPU events.
- `submit_ns` and `wait_ns` split host-observed time into pre-fence dispatch vs post-dispatch waiting.
- `scheduling_latency_ns` approximates overhead outside GPU execution (queueing, launch, synchronization, runtime overhead).
- On non-GPU captures, focus on `duration_ns`, `submit_ns`, and `wait_ns`; GPU-related fields remain empty.
