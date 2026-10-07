// Structured logging: one JSON object per line on stderr, e.g.
// {"ts_ms":12,"level":"info","event":"node_finished","node":"fp32","ms":8.1}
#pragma once

#include <initializer_list>
#include <string>
#include <utility>

namespace eio {

enum class LogLevel { Debug = 0, Info = 1, Warning = 2, Error = 3, Off = 4 };

void set_log_level(LogLevel level);
LogLevel log_level();

// Fields are key/value pairs; values are written as JSON strings.
using LogFields = std::initializer_list<std::pair<const char*, std::string>>;
void log(LogLevel level, const char* event, LogFields fields = {});

}  // namespace eio
