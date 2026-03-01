#include <kokkos_hotspot/tool_common.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace tool_common {

bool
env_flag(const char* name)
{
  const char* val = std::getenv(name);
  return val && val[0] && std::strcmp(val, "0") != 0;
}

std::string
get_env(const char* name, const char* fallback)
{
  const char* val = std::getenv(name);
  if (val && val[0]) {
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
debug_log(const char* env_name, const char* tag, const char* msg)
{
  if (!env_flag(env_name)) {
    return;
  }
  std::fprintf(stderr, "[%s] %s\n", tag, msg);
}

}  // namespace tool_common
