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

bool
env_flag(const char* name)
{
  if (const char* val = std::getenv(name); val && val[0]) {
    return std::strcmp(val, "0") != 0;
  }
  return false;
}

std::string
get_env(const char* name, const char* fallback)
{
  if (const char* val = std::getenv(name); val && val[0]) {
    return std::string(val);
  }
  return std::string(fallback ? fallback : "");
}

bool
file_exists(const std::string& path)
{
  std::ifstream in(path);
  return in.good();
}

std::string
csv_escape(std::string_view input)
{
  bool needs_quotes = false;
  for (char c : input) {
    if (c == ',' || c == '"' || c == '\n' || c == '\r') {
      needs_quotes = true;
      break;
    }
  }
  if (!needs_quotes) {
    return std::string(input);
  }
  std::string out;
  out.reserve(input.size() + 2);
  out.push_back('"');
  for (char c : input) {
    if (c == '"') {
      out.push_back('"');
    }
    out.push_back(c);
  }
  out.push_back('"');
  return out;
}

void
stderr_println(std::string_view msg)
{
#if defined(__cpp_lib_print)
  std::print(stderr, "{}\n", msg);
#else
  std::fwrite(msg.data(), sizeof(char), msg.size(), stderr);
  std::fputc('\n', stderr);
#endif
}

void
debug_log(const char* env_name, const char* tag, const char* msg)
{
  if (!env_flag(env_name)) {
    return;
  }
  std::string line = "[";
  line += tag;
  line += "] ";
  line += msg;
  stderr_println(line);
}

}  // namespace tool_common
