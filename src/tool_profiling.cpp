#include <impl/Kokkos_Profiling_C_Interface.h>

#include <Kokkos_Core.hpp>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
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

static auto
gpu_event_create(GpuEvent* event) -> GpuError
{
  return cudaEventCreate(event);
}

static auto
gpu_event_destroy(GpuEvent event) -> GpuError
{
  return cudaEventDestroy(event);
}

static auto
gpu_event_record(GpuEvent event) -> GpuError
{
  return cudaEventRecord(event, nullptr);
}

static auto
gpu_event_elapsed_time(float* milliseconds, GpuEvent start,
                       GpuEvent end) -> GpuError
{
  return cudaEventElapsedTime(milliseconds, start, end);
}

static auto
gpu_event_synchronize(GpuEvent event) -> GpuError
{
  return cudaEventSynchronize(event);
}

static auto
gpu_error_string(GpuError err) -> const char*
{
  return cudaGetErrorString(err);
}

static auto
gpu_runtime_ready() -> bool
{
  int device_count = 0;
  if (cudaError_t count_err = cudaGetDeviceCount(&device_count);
      count_err != cudaSuccess || device_count <= 0) {
    cudaGetLastError();
    return false;
  }
  if (cudaError_t init_err = cudaFree(nullptr); init_err != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return true;
}
#else
using GpuEvent = hipEvent_t;
using GpuError = hipError_t;
static constexpr GpuError gpu_success = hipSuccess;

static auto
gpu_event_create(GpuEvent* event) -> GpuError
{
  return hipEventCreate(event);
}

static auto
gpu_event_destroy(GpuEvent event) -> GpuError
{
  return hipEventDestroy(event);
}

static auto
gpu_event_record(GpuEvent event) -> GpuError
{
  return hipEventRecord(event, nullptr);
}

static auto
gpu_event_elapsed_time(float* milliseconds, GpuEvent start,
                       GpuEvent end) -> GpuError
{
  return hipEventElapsedTime(milliseconds, start, end);
}

static auto
gpu_event_synchronize(GpuEvent event) -> GpuError
{
  return hipEventSynchronize(event);
}

static auto
gpu_error_string(GpuError err) -> const char*
{
  return hipGetErrorString(err);
}

static auto
gpu_runtime_ready() -> bool
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
  std::mutex pool_mutex{};
  std::vector<GpuEvent> event_pool{};
};

static GpuTimingContext
    gpu_timing;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
#endif

struct KernelRecord {
  std::string name{};
  std::string type{};
  TimePoint begin{};
  std::thread::id thread_id;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  GpuEventPair gpu_events{};
#endif
};

struct RecordIds {
  uint64_t sequence_id = 0;
  uint64_t kernel_id = 0;
};

struct RecordTimes {
  TimePoint dispatch_end{};
  TimePoint end{};
};

static std::atomic<uint64_t>
    next_kid{1};  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
static std::atomic<uint64_t>
    next_seq{0};  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
static std::mutex
    active_mutex;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
static std::unordered_map<uint64_t, KernelRecord>
    active_records;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

static std::mutex
    out_mutex;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
static std::ofstream
    out_file;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
static bool
    out_ready = false;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
static bool
    flush_each = false;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
static std::string
    out_path;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
static TimePoint
    start_time;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
static std::once_flag
    out_once;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

static auto
debug_enabled() -> bool
{
  return tool_common::env_flag("KOKKOS_TIMING_DEBUG");
}

static auto
debug_log(std::string_view message) -> void
{
  tool_common::debug_log("KOKKOS_TIMING_DEBUG", "kokkos_profiling", message);
}

static auto
get_env(const char* name, std::string_view fallback) -> std::string
{
  return tool_common::get_env(name, fallback);
}

static auto
env_flag(const char* name) -> bool
{
  return tool_common::env_flag(name);
}

static auto
file_exists(const std::string& path) -> bool
{
  return tool_common::file_exists(path);
}

static auto
csv_escape(std::string_view input) -> std::string
{
  return tool_common::csv_escape(input);
}

static auto
gpu_events_requested() -> bool
{
  const char* raw = std::getenv("KOKKOS_TIMING_GPU_EVENTS");
  if ((raw == nullptr) || (*raw == '\0')) {
    return true;
  }
  return std::strcmp(raw, "0") != 0;
}

static auto
ms_to_ns(float milliseconds) -> int64_t
{
  if (!std::isfinite(milliseconds) || milliseconds < 0.0F) {
    return -1;
  }
  const long double nanoseconds =
      static_cast<long double>(milliseconds) * 1.0e6L;
  if (nanoseconds >
      static_cast<long double>(std::numeric_limits<int64_t>::max())) {
    return -1;
  }
  return static_cast<int64_t>(std::llround(nanoseconds));
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
static auto
gpu_debug_error(const char* what, GpuError err) -> void
{
  if (!debug_enabled()) {
    return;
  }
  std::string msg = "[kokkos_profiling] ";
  msg += what;
  msg += " failed: ";
  msg += gpu_error_string(err);
  tool_common::stderr_println(msg);
}

static auto
gpu_init() -> bool
{
  if (!gpu_timing.requested) {
    return false;
  }
  if (!gpu_runtime_ready()) {
    if (debug_enabled()) {
      tool_common::stderr_println(
          "[kokkos_profiling] no GPU runtime/device available");
    }
    return false;
  }
  if (GpuError create_err = gpu_event_create(&gpu_timing.origin);
      create_err != gpu_success) {
    gpu_debug_error("event create (origin)", create_err);
    return false;
  }
  if (GpuError record_err = gpu_event_record(gpu_timing.origin);
      record_err != gpu_success) {
    gpu_debug_error("event record (origin)", record_err);
    gpu_event_destroy(gpu_timing.origin);
    return false;
  }
  gpu_timing.origin_ready = true;
  gpu_timing.enabled = true;
  return true;
}

static auto
gpu_acquire_event(GpuEvent* event) -> bool
{
  {
    std::scoped_lock lock(gpu_timing.pool_mutex);
    if (!gpu_timing.event_pool.empty()) {
      *event = gpu_timing.event_pool.back();
      gpu_timing.event_pool.pop_back();
      return true;
    }
  }
  if (GpuError create_err = gpu_event_create(event);
      create_err != gpu_success) {
    gpu_debug_error("event create", create_err);
    return false;
  }
  return true;
}

static auto
gpu_release_event(GpuEvent event) -> void
{
  std::scoped_lock lock(gpu_timing.pool_mutex);
  gpu_timing.event_pool.push_back(event);
}

static auto
gpu_release_event_pair(GpuEventPair& events) -> void
{
  if (!events.valid) {
    return;
  }
  gpu_release_event(events.begin);
  gpu_release_event(events.end);
  events.valid = false;
}

static auto
gpu_acquire_event_pair() -> GpuEventPair
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

static auto
gpu_collect_sample(const GpuEventPair& events) -> GpuTimingSample
{
  GpuTimingSample sample;
  if (!gpu_timing.enabled || !events.valid) {
    return sample;
  }

  if (GpuError sync_err = gpu_event_synchronize(events.end);
      sync_err != gpu_success) {
    gpu_debug_error("event synchronize (end)", sync_err);
    return sample;
  }

  float gpu_duration_ms = 0.0F;
  if (GpuError duration_err =
          gpu_event_elapsed_time(&gpu_duration_ms, events.begin, events.end);
      duration_err != gpu_success) {
    gpu_debug_error("event elapsed (duration)", duration_err);
    return sample;
  }
  sample.duration_ns = ms_to_ns(gpu_duration_ms);

  if (!gpu_timing.origin_ready) {
    return sample;
  }

  float gpu_begin_ms = 0.0F;
  if (GpuError begin_err =
          gpu_event_elapsed_time(&gpu_begin_ms, gpu_timing.origin, events.begin);
      begin_err == gpu_success) {
    sample.begin_ns = ms_to_ns(gpu_begin_ms);
  } else {
    gpu_debug_error("event elapsed (begin)", begin_err);
  }

  float gpu_end_ms = 0.0F;
  if (GpuError end_err =
          gpu_event_elapsed_time(&gpu_end_ms, gpu_timing.origin, events.end);
      end_err == gpu_success) {
    sample.end_ns = ms_to_ns(gpu_end_ms);
  } else {
    gpu_debug_error("event elapsed (end)", end_err);
  }
  return sample;
}

static auto
gpu_shutdown() -> void
{
  std::vector<GpuEvent> pool;
  {
    std::scoped_lock lock(gpu_timing.pool_mutex);
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

static auto
initialize_output_once() -> void
{
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
    tool_common::stderr_println(
        "[kokkos_profiling] KOKKOS_TIMING_GPU_EVENTS requested but backend "
        "has no GPU support");
  }
#endif

  bool write_header = true;
  if (append && file_exists(out_path)) {
    write_header = false;
  }

  std::ios::openmode mode = std::ios::out | std::ios::trunc;
  if (append) {
    mode = std::ios::out | std::ios::app;
  }

  out_file.open(out_path, mode);
  if (!out_file.is_open()) {
    if (debug_enabled()) {
      tool_common::stderr_println(
          std::string("[kokkos_profiling] unable to open ") + out_path);
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
}

static auto
open_output() -> void
{
  std::call_once(out_once, initialize_output_once);
}

static auto
write_optional_int(std::ofstream& out, int64_t value) -> void
{
  if (value >= 0) {
    out << value;
  }
}

static auto
non_negative_diff(int64_t lhs, int64_t rhs) -> int64_t
{
  return lhs >= rhs ? (lhs - rhs) : 0;
}

static auto
write_record(
    const RecordIds& ids, const KernelRecord& record,
    const RecordTimes& times,
    const GpuTimingSample& gpu_sample) -> void
{
  open_output();
  if (!out_ready) {
    return;
  }

  int64_t begin_ns = 0;
  begin_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(record.begin -
                                                           start_time)
          .count();
  int64_t dispatch_end_ns = 0;
  dispatch_end_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          times.dispatch_end - start_time)
          .count();
  int64_t end_ns = 0;
  end_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(times.end - start_time)
          .count();
  const int64_t duration_ns = non_negative_diff(end_ns, begin_ns);
  const int64_t submit_ns = non_negative_diff(dispatch_end_ns, begin_ns);
  const int64_t wait_ns = non_negative_diff(end_ns, dispatch_end_ns);
  const int64_t scheduling_latency_ns =
      gpu_sample.duration_ns >= 0
          ? non_negative_diff(duration_ns, gpu_sample.duration_ns)
          : -1;
  const auto thread_hash = std::hash<std::thread::id>{}(record.thread_id);

  std::scoped_lock lock(out_mutex);
  out_file << ids.sequence_id << ',' << ids.kernel_id << ','
           << csv_escape(record.name)
           << ','
           << csv_escape(record.type) << ',' << thread_hash << ',' << begin_ns
           << ',' << end_ns << ',' << duration_ns << ',' << dispatch_end_ns
           << ',' << submit_ns << ',' << wait_ns << ',';
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

static auto
begin_kernel(const char* kernel_name, std::string_view kernel_type,
             uint64_t* kernel_id_ptr) -> void
{
  open_output();
  if (kernel_id_ptr == nullptr) {
    debug_log("missing kID pointer");
    return;
  }
  uint64_t kernel_id = 0;
  kernel_id = next_kid.fetch_add(1, std::memory_order_relaxed);
  *kernel_id_ptr = kernel_id;

  KernelRecord record;
  record.name = (kernel_name != nullptr) ? kernel_name : "";
  record.type.assign(kernel_type);
  record.begin = std::chrono::steady_clock::now();
  record.thread_id = std::this_thread::get_id();

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  if (gpu_timing.enabled) {
    record.gpu_events = gpu_acquire_event_pair();
    if (record.gpu_events.valid) {
      if (GpuError record_err = gpu_event_record(record.gpu_events.begin);
          record_err != gpu_success) {
        gpu_debug_error("event record (begin)", record_err);
        gpu_release_event_pair(record.gpu_events);
      }
    }
  }
#endif

  std::scoped_lock lock(active_mutex);
  active_records[kernel_id] = std::move(record);
}

static auto
end_kernel(const char* kernel_type, uint64_t kernel_id) -> void
{
  KernelRecord record;
  {
    std::scoped_lock lock(active_mutex);
    auto record_it = active_records.find(kernel_id);
    if (record_it == active_records.end()) {
      return;
    }
    record = std::move(record_it->second);
    active_records.erase(record_it);
  }

  if (record.type.empty() && (kernel_type != nullptr)) {
    record.type = kernel_type;
  }

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  if (gpu_timing.enabled && record.gpu_events.valid) {
    if (GpuError record_err = gpu_event_record(record.gpu_events.end);
        record_err != gpu_success) {
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

  uint64_t sequence_id = 0;
  sequence_id = next_seq.fetch_add(1, std::memory_order_relaxed) + 1;
  const RecordIds ids{sequence_id, kernel_id};
  const RecordTimes times{dispatch_end_time, end_time};
  write_record(
      ids, record, times, gpu_sample);
}

static auto
release_active_records() -> void
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  std::unordered_map<uint64_t, KernelRecord> pending;
  {
    std::scoped_lock lock(active_mutex);
    pending.swap(active_records);
  }
  for (auto& entry : pending) {
    if (entry.second.gpu_events.valid) {
      gpu_release_event_pair(entry.second.gpu_events);
    }
  }
#else
  std::scoped_lock lock(active_mutex);
  active_records.clear();
#endif
}

extern "C" void
kokkosp_init_library(
    int load_seq, uint64_t interface_ver,  // NOLINT(bugprone-easily-swappable-parameters)
    uint32_t dev_info_count,
    Kokkos_Profiling_KokkosPDeviceInfo* device_info)
{
  static_cast<void>(load_seq);
  static_cast<void>(interface_ver);
  static_cast<void>(dev_info_count);
  static_cast<void>(device_info);
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
  std::scoped_lock lock(out_mutex);
  out_file.flush();
  out_file.close();
}

extern "C" void
kokkosp_begin_parallel_for(
    const char* name, const uint32_t dev_info,
    uint64_t* kID)  // NOLINT(readability-non-const-parameter)
{
  static_cast<void>(dev_info);
  begin_kernel(name, "for", kID);
}

extern "C" void
kokkosp_end_parallel_for(const uint64_t kID)
{
  end_kernel("for", kID);
}

extern "C" void
kokkosp_begin_parallel_reduce(
    const char* name, const uint32_t dev_info,
    uint64_t* kID)  // NOLINT(readability-non-const-parameter)
{
  static_cast<void>(dev_info);
  begin_kernel(name, "reduce", kID);
}

extern "C" void
kokkosp_end_parallel_reduce(const uint64_t kID)
{
  end_kernel("reduce", kID);
}

extern "C" void
kokkosp_begin_parallel_scan(
    const char* name, const uint32_t dev_info,
    uint64_t* kID)  // NOLINT(readability-non-const-parameter)
{
  static_cast<void>(dev_info);
  begin_kernel(name, "scan", kID);
}

extern "C" void
kokkosp_end_parallel_scan(const uint64_t kID)
{
  end_kernel("scan", kID);
}
