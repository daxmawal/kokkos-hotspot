#include <kokkos_hotspot/tool_common.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#if defined(__has_include)
#if __has_include(<print>)
#include <print>
#endif
#endif

namespace tool_common {

auto
env_flag(const char* name) -> bool
{
  if (const char* val = std::getenv(name);
      (val != nullptr) && (*val != '\0')) {
    return std::strcmp(val, "0") != 0;
  }
  return false;
}

auto
get_env(const char* name, std::string_view fallback) -> std::string
{
  if (const char* val = std::getenv(name);
      (val != nullptr) && (*val != '\0')) {
    return {val};
  }
  return {fallback.begin(), fallback.end()};
}

auto
file_exists(const std::string& path) -> bool
{
  std::ifstream input_stream(path);
  return input_stream.good();
}

auto
csv_escape(std::string_view input) -> std::string
{
  bool needs_quotes = false;
  for (char chr : input) {
    if (chr == ',' || chr == '"' || chr == '\n' || chr == '\r') {
      needs_quotes = true;
      break;
    }
  }
  if (!needs_quotes) {
    return {input.begin(), input.end()};
  }
  std::string out;
  out.reserve(input.size() + 2);
  out.push_back('"');
  for (char chr : input) {
    if (chr == '"') {
      out.push_back('"');
    }
    out.push_back(chr);
  }
  out.push_back('"');
  return out;
}

auto
stderr_println(std::string_view msg) -> void
{
#if defined(__cpp_lib_print)
  std::print(stderr, "{}\n", msg);
#else
  std::fwrite(msg.data(), sizeof(char), msg.size(), stderr);
  std::fputc('\n', stderr);
#endif
}

auto
debug_log(const char* env_name, std::string_view tag,
          std::string_view msg) -> void
{
  if (!env_flag(env_name)) {
    return;
  }
  std::string line = "[";
  line.append(tag);
  line += "] ";
  line.append(msg);
  stderr_println(line);
}

}  // namespace tool_common
