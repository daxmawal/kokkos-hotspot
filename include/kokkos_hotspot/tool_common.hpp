#pragma once

#include <string>
#include <string_view>

namespace tool_common {

auto env_flag(const char* name) -> bool;
auto get_env(const char* name, std::string_view fallback) -> std::string;
auto file_exists(const std::string& path) -> bool;
auto csv_escape(std::string_view input) -> std::string;
auto stderr_println(std::string_view msg) -> void;
auto debug_log(const char* env_name, std::string_view tag,
               std::string_view msg) -> void;

}  // namespace tool_common
