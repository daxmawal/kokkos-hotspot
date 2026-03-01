#pragma once

#include <string>
#include <string_view>

namespace tool_common {

bool env_flag(const char* name);
std::string get_env(const char* name, const char* fallback);
bool file_exists(const std::string& path);
std::string csv_escape(std::string_view input);
void debug_log(const char* env_name, const char* tag, const char* msg);

}  // namespace tool_common
