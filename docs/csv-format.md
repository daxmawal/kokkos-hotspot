# CSV Format Reference

The profiling tool writes one row per kernel event with this header:

```text
seq,kid,name,type,thread_id,begin_ns,end_ns,duration_ns,dispatch_end_ns,submit_ns,wait_ns,gpu_begin_ns,gpu_end_ns,gpu_duration_ns,scheduling_latency_ns
```

Key fields:

- `name`: kernel label
- `type`: event type (`for`, `reduce`, `scan`)
- `duration_ns`: synchronized CPU duration
- `gpu_duration_ns`: CUDA/HIP GPU-event duration when available
- `scheduling_latency_ns`: `max(duration_ns - gpu_duration_ns, 0)`

Timing units are nanoseconds for all `*_ns` fields.
