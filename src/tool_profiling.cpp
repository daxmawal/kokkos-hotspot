#include <impl/Kokkos_Profiling_C_Interface.h>

#include <Kokkos_Core.hpp>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <kokkos_hotspot/tool_common.hpp>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif
#if defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

using TimePoint = std::chrono::steady_clock::time_point;

struct GpuTimingSample {
  int64_t begin_ns = -1;
  int64_t end_ns = -1;
  int64_t duration_ns = -1;
};

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)

#if defined(KOKKOS_ENABLE_CUDA)
using GpuEvent = cudaEvent_t;
using GpuError = cudaError_t;
static constexpr GpuError gpu_success = cudaSuccess;

static GpuError
gpu_event_create(GpuEvent* event)
{
  return cudaEventCreate(event);
}

static GpuError
gpu_event_destroy(GpuEvent event)
{
  return cudaEventDestroy(event);
}

static GpuError
gpu_event_record(GpuEvent event)
{
  return cudaEventRecord(event, 0);
}

static GpuError
gpu_event_elapsed_time(float* milliseconds, GpuEvent start, GpuEvent end)
{
  return cudaEventElapsedTime(milliseconds, start, end);
}

static GpuError
gpu_event_synchronize(GpuEvent event)
{
  return cudaEventSynchronize(event);
}

static const char*
gpu_error_string(GpuError err)
{
  return cudaGetErrorString(err);
}

static bool
gpu_runtime_ready()
{
  int device_count = 0;
  cudaError_t count_err = cudaGetDeviceCount(&device_count);
  if (count_err != cudaSuccess || device_count <= 0) {
    cudaGetLastError();
    return false;
  }
  cudaError_t init_err = cudaFree(0);
  if (init_err != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return true;
}
#else
using GpuEvent = hipEvent_t;
using GpuError = hipError_t;
static constexpr GpuError gpu_success = hipSuccess;

static GpuError
gpu_event_create(GpuEvent* event)
{
  return hipEventCreate(event);
}

static GpuError
gpu_event_destroy(GpuEvent event)
{
  return hipEventDestroy(event);
}

static GpuError
gpu_event_record(GpuEvent event)
{
  return hipEventRecord(event, nullptr);
}

static GpuError
gpu_event_elapsed_time(float* milliseconds, GpuEvent start, GpuEvent end)
{
  return hipEventElapsedTime(milliseconds, start, end);
}

static GpuError
gpu_event_synchronize(GpuEvent event)
{
  return hipEventSynchronize(event);
}

static const char*
gpu_error_string(GpuError err)
{
  return hipGetErrorString(err);
}

static bool
gpu_runtime_ready()
{
  int device_count = 0;
  hipError_t count_err = hipGetDeviceCount(&device_count);
  return count_err == hipSuccess && device_count > 0;
}
#endif

struct GpuEventPair {
  bool valid = false;
  GpuEvent begin{};
  GpuEvent end{};
};

struct GpuTimingContext {
  bool requested = false;
  bool enabled = false;
  bool origin_ready = false;
  GpuEvent origin{};
  std::mutex pool_mutex;
  std::vector<GpuEvent> event_pool;
};

static GpuTimingContext gpu_timing;
#endif

struct KernelRecord {
  std::string name;
  std::string type;
  TimePoint begin;
  std::thread::id thread_id;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  GpuEventPair gpu_events;
#endif
};

static std::atomic<uint64_t> next_kid{1};
static std::atomic<uint64_t> next_seq{0};
static std::mutex active_mutex;
static std::unordered_map<uint64_t, KernelRecord> active_records;

static std::mutex out_mutex;
static std::ofstream out_file;
static bool out_ready = false;
static bool flush_each = false;
static std::string out_path;
static TimePoint start_time;
static std::once_flag out_once;

static bool
debug_enabled()
{
  return tool_common::env_flag("KOKKOS_TIMING_DEBUG");
}

static void
debug_log(const char* msg)
{
  tool_common::debug_log("KOKKOS_TIMING_DEBUG", "kokkos_profiling", msg);
}

static std::string
get_env(const char* name, const char* fallback)
{
  return tool_common::get_env(name, fallback);
}

static bool
env_flag(const char* name)
{
  return tool_common::env_flag(name);
}

static bool
file_exists(const std::string& path)
{
  return tool_common::file_exists(path);
}

static std::string
csv_escape(std::string_view input)
{
  return tool_common::csv_escape(input);
}

static bool
gpu_events_requested()
{
  const char* raw = std::getenv("KOKKOS_TIMING_GPU_EVENTS");
  if (!raw || !raw[0]) {
    return true;
  }
  return env_flag("KOKKOS_TIMING_GPU_EVENTS");
}

static int64_t
ms_to_ns(float ms)
{
  if (!std::isfinite(ms) || ms < 0.0f) {
    return -1;
  }
  const long double ns = static_cast<long double>(ms) * 1.0e6L;
  if (ns > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
    return -1;
  }
  return static_cast<int64_t>(ns + 0.5L);
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
static void
gpu_debug_error(const char* what, GpuError err)
{
  if (!debug_enabled()) {
    return;
  }
  std::fprintf(
      stderr, "[kokkos_profiling] %s failed: %s\n", what,
      gpu_error_string(err));
}

static bool
gpu_init()
{
  if (!gpu_timing.requested) {
    return false;
  }
  if (!gpu_runtime_ready()) {
    if (debug_enabled()) {
      std::fprintf(
          stderr, "[kokkos_profiling] no GPU runtime/device available\n");
    }
    return false;
  }
  GpuError create_err = gpu_event_create(&gpu_timing.origin);
  if (create_err != gpu_success) {
    gpu_debug_error("event create (origin)", create_err);
    return false;
  }
  GpuError record_err = gpu_event_record(gpu_timing.origin);
  if (record_err != gpu_success) {
    gpu_debug_error("event record (origin)", record_err);
    gpu_event_destroy(gpu_timing.origin);
    return false;
  }
  gpu_timing.origin_ready = true;
  gpu_timing.enabled = true;
  return true;
}

static bool
gpu_acquire_event(GpuEvent* event)
{
  {
    std::lock_guard<std::mutex> lock(gpu_timing.pool_mutex);
    if (!gpu_timing.event_pool.empty()) {
      *event = gpu_timing.event_pool.back();
      gpu_timing.event_pool.pop_back();
      return true;
    }
  }
  GpuError create_err = gpu_event_create(event);
  if (create_err != gpu_success) {
    gpu_debug_error("event create", create_err);
    return false;
  }
  return true;
}

static void
gpu_release_event(GpuEvent event)
{
  std::lock_guard<std::mutex> lock(gpu_timing.pool_mutex);
  gpu_timing.event_pool.push_back(event);
}

static void
gpu_release_event_pair(GpuEventPair& events)
{
  if (!events.valid) {
    return;
  }
  gpu_release_event(events.begin);
  gpu_release_event(events.end);
  events.valid = false;
}

static GpuEventPair
gpu_acquire_event_pair()
{
  GpuEventPair events;
  if (!gpu_timing.enabled) {
    return events;
  }
  if (!gpu_acquire_event(&events.begin)) {
    return events;
  }
  if (!gpu_acquire_event(&events.end)) {
    gpu_release_event(events.begin);
    return events;
  }
  events.valid = true;
  return events;
}

static GpuTimingSample
gpu_collect_sample(const GpuEventPair& events)
{
  GpuTimingSample sample;
  if (!gpu_timing.enabled || !events.valid) {
    return sample;
  }

  GpuError sync_err = gpu_event_synchronize(events.end);
  if (sync_err != gpu_success) {
    gpu_debug_error("event synchronize (end)", sync_err);
    return sample;
  }

  float gpu_duration_ms = 0.0f;
  GpuError duration_err =
      gpu_event_elapsed_time(&gpu_duration_ms, events.begin, events.end);
  if (duration_err != gpu_success) {
    gpu_debug_error("event elapsed (duration)", duration_err);
    return sample;
  }
  sample.duration_ns = ms_to_ns(gpu_duration_ms);

  if (!gpu_timing.origin_ready) {
    return sample;
  }

  float gpu_begin_ms = 0.0f;
  GpuError begin_err =
      gpu_event_elapsed_time(&gpu_begin_ms, gpu_timing.origin, events.begin);
  if (begin_err == gpu_success) {
    sample.begin_ns = ms_to_ns(gpu_begin_ms);
  } else {
    gpu_debug_error("event elapsed (begin)", begin_err);
  }

  float gpu_end_ms = 0.0f;
  GpuError end_err =
      gpu_event_elapsed_time(&gpu_end_ms, gpu_timing.origin, events.end);
  if (end_err == gpu_success) {
    sample.end_ns = ms_to_ns(gpu_end_ms);
  } else {
    gpu_debug_error("event elapsed (end)", end_err);
  }
  return sample;
}

static void
gpu_shutdown()
{
  std::vector<GpuEvent> pool;
  {
    std::lock_guard<std::mutex> lock(gpu_timing.pool_mutex);
    pool.swap(gpu_timing.event_pool);
  }
  for (GpuEvent event : pool) {
    GpuError destroy_err = gpu_event_destroy(event);
    if (destroy_err != gpu_success) {
      gpu_debug_error("event destroy (pool)", destroy_err);
    }
  }

  if (gpu_timing.origin_ready) {
    GpuError destroy_err = gpu_event_destroy(gpu_timing.origin);
    if (destroy_err != gpu_success) {
      gpu_debug_error("event destroy (origin)", destroy_err);
    }
  }

  gpu_timing.enabled = false;
  gpu_timing.origin_ready = false;
}
#endif

static void
open_output()
{
  std::call_once(out_once, [] {
    start_time = std::chrono::steady_clock::now();
    out_path = get_env("KOKKOS_TIMING_OUT", "kokkos_kernel_times.csv");
    const bool append = env_flag("KOKKOS_TIMING_APPEND");
    flush_each = env_flag("KOKKOS_TIMING_FLUSH");

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    gpu_timing.requested = gpu_events_requested();
    if (gpu_timing.requested) {
      gpu_init();
    }
#else
    if (gpu_events_requested() && debug_enabled()) {
      std::fprintf(
          stderr,
          "[kokkos_profiling] KOKKOS_TIMING_GPU_EVENTS requested but backend "
          "has no GPU support\n");
    }
#endif

    bool write_header = true;
    if (append && file_exists(out_path)) {
      write_header = false;
    }

    std::ios::openmode mode = std::ios::out;
    if (append) {
      mode |= std::ios::app;
    } else {
      mode |= std::ios::trunc;
    }

    out_file.open(out_path, mode);
    if (!out_file.is_open()) {
      if (debug_enabled()) {
        std::fprintf(
            stderr, "[kokkos_profiling] unable to open %s\n", out_path.c_str());
      }
      return;
    }

    out_ready = true;
    if (write_header) {
      out_file << "seq,kid,name,type,thread_id,begin_ns,end_ns,duration_ns,"
                  "dispatch_end_ns,submit_ns,wait_ns,gpu_begin_ns,gpu_end_ns,"
                  "gpu_duration_ns,scheduling_latency_ns\n";
      if (flush_each) {
        out_file.flush();
      }
    }
  });
}

static void
write_optional_int(std::ofstream& out, int64_t value)
{
  if (value >= 0) {
    out << value;
  }
}

static int64_t
non_negative_diff(int64_t lhs, int64_t rhs)
{
  return lhs >= rhs ? (lhs - rhs) : 0;
}

static void
write_record(
    uint64_t seq, uint64_t kid, std::string_view name, std::string_view type,
    std::thread::id thread_id, TimePoint begin, TimePoint dispatch_end,
    TimePoint end, const GpuTimingSample& gpu_sample)
{
  open_output();
  if (!out_ready) {
    return;
  }

  const int64_t begin_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(begin - start_time)
          .count();
  const int64_t dispatch_end_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          dispatch_end - start_time)
          .count();
  const int64_t end_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_time)
          .count();
  const int64_t duration_ns = non_negative_diff(end_ns, begin_ns);
  const int64_t submit_ns = non_negative_diff(dispatch_end_ns, begin_ns);
  const int64_t wait_ns = non_negative_diff(end_ns, dispatch_end_ns);
  const int64_t scheduling_latency_ns =
      gpu_sample.duration_ns >= 0
          ? non_negative_diff(duration_ns, gpu_sample.duration_ns)
          : -1;
  const uint64_t thread_hash =
      static_cast<uint64_t>(std::hash<std::thread::id>{}(thread_id));

  std::lock_guard<std::mutex> lock(out_mutex);
  out_file << seq << ',' << kid << ',' << csv_escape(name) << ','
           << csv_escape(type) << ',' << thread_hash << ',' << begin_ns << ','
           << end_ns << ',' << duration_ns << ',' << dispatch_end_ns << ','
           << submit_ns << ',' << wait_ns << ',';
  write_optional_int(out_file, gpu_sample.begin_ns);
  out_file << ',';
  write_optional_int(out_file, gpu_sample.end_ns);
  out_file << ',';
  write_optional_int(out_file, gpu_sample.duration_ns);
  out_file << ',';
  write_optional_int(out_file, scheduling_latency_ns);
  out_file << '\n';
  if (flush_each) {
    out_file.flush();
  }
}

static void
begin_kernel(const char* name, const char* type, uint64_t* kID)
{
  open_output();
  if (!kID) {
    debug_log("missing kID pointer");
    return;
  }
  const uint64_t kid = next_kid.fetch_add(1, std::memory_order_relaxed);
  *kID = kid;

  KernelRecord record;
  record.name = name ? name : "";
  record.type = type ? type : "";
  record.begin = std::chrono::steady_clock::now();
  record.thread_id = std::this_thread::get_id();

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  if (gpu_timing.enabled) {
    record.gpu_events = gpu_acquire_event_pair();
    if (record.gpu_events.valid) {
      GpuError record_err = gpu_event_record(record.gpu_events.begin);
      if (record_err != gpu_success) {
        gpu_debug_error("event record (begin)", record_err);
        gpu_release_event_pair(record.gpu_events);
      }
    }
  }
#endif

  std::lock_guard<std::mutex> lock(active_mutex);
  active_records[kid] = std::move(record);
}

static void
end_kernel(const char* type, const uint64_t kID)
{
  KernelRecord record;
  {
    std::lock_guard<std::mutex> lock(active_mutex);
    auto it = active_records.find(kID);
    if (it == active_records.end()) {
      return;
    }
    record = std::move(it->second);
    active_records.erase(it);
  }

  if (record.type.empty() && type) {
    record.type = type;
  }

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  if (gpu_timing.enabled && record.gpu_events.valid) {
    GpuError record_err = gpu_event_record(record.gpu_events.end);
    if (record_err != gpu_success) {
      gpu_debug_error("event record (end)", record_err);
      gpu_release_event_pair(record.gpu_events);
    }
  }
#endif

  const auto dispatch_end_time = std::chrono::steady_clock::now();
  Kokkos::fence("kokkos_profiling");
  const auto end_time = std::chrono::steady_clock::now();

  GpuTimingSample gpu_sample;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  if (gpu_timing.enabled && record.gpu_events.valid) {
    gpu_sample = gpu_collect_sample(record.gpu_events);
    gpu_release_event_pair(record.gpu_events);
  }
#endif

  const uint64_t seq = next_seq.fetch_add(1, std::memory_order_relaxed) + 1;
  write_record(
      seq, kID, record.name, record.type, record.thread_id, record.begin,
      dispatch_end_time, end_time, gpu_sample);
}

static void
release_active_records()
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  std::unordered_map<uint64_t, KernelRecord> pending;
  {
    std::lock_guard<std::mutex> lock(active_mutex);
    pending.swap(active_records);
  }
  for (auto& entry : pending) {
    if (entry.second.gpu_events.valid) {
      gpu_release_event_pair(entry.second.gpu_events);
    }
  }
#else
  std::lock_guard<std::mutex> lock(active_mutex);
  active_records.clear();
#endif
}

extern "C" void
kokkosp_init_library(
    int, uint64_t, uint32_t, Kokkos_Profiling_KokkosPDeviceInfo*)
{
  open_output();
  debug_log("init");
}

extern "C" void
kokkosp_finalize_library()
{
  debug_log("finalize");
  release_active_records();
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  gpu_shutdown();
#endif
  if (!out_ready) {
    return;
  }
  std::lock_guard<std::mutex> lock(out_mutex);
  out_file.flush();
  out_file.close();
}

extern "C" void
kokkosp_begin_parallel_for(const char* name, const uint32_t, uint64_t* kID)
{
  begin_kernel(name, "for", kID);
}

extern "C" void
kokkosp_end_parallel_for(const uint64_t kID)
{
  end_kernel("for", kID);
}

extern "C" void
kokkosp_begin_parallel_reduce(const char* name, const uint32_t, uint64_t* kID)
{
  begin_kernel(name, "reduce", kID);
}

extern "C" void
kokkosp_end_parallel_reduce(const uint64_t kID)
{
  end_kernel("reduce", kID);
}

extern "C" void
kokkosp_begin_parallel_scan(const char* name, const uint32_t, uint64_t* kID)
{
  begin_kernel(name, "scan", kID);
}

extern "C" void
kokkosp_end_parallel_scan(const uint64_t kID)
{
  end_kernel("scan", kID);
}
