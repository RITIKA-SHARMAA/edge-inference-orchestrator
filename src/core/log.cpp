#include "log.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>

#include <nlohmann/json.hpp>

namespace eio {

namespace {
// Default comes from EIO_LOG_LEVEL (0-4) if set, otherwise warnings and errors only.
LogLevel initial_level() {
  if (const char* env = std::getenv("EIO_LOG_LEVEL")) {
    if (env[0] >= '0' && env[0] <= '4' && env[1] == '\0') {
      return static_cast<LogLevel>(env[0] - '0');
    }
  }
  return LogLevel::Warning;
}

std::atomic<int> g_level{static_cast<int>(initial_level())};
std::mutex g_write_mutex;  // keeps lines from different threads from interleaving
const auto g_start = std::chrono::steady_clock::now();

const char* level_name(LogLevel level) {
  switch (level) {
    case LogLevel::Debug:
      return "debug";
    case LogLevel::Info:
      return "info";
    case LogLevel::Warning:
      return "warning";
    case LogLevel::Error:
      return "error";
    case LogLevel::Off:
      break;
  }
  return "off";
}
}  // namespace

void set_log_level(LogLevel level) { g_level.store(static_cast<int>(level)); }
LogLevel log_level() { return static_cast<LogLevel>(g_level.load()); }

void log(LogLevel level, const char* event, LogFields fields) {
  if (level == LogLevel::Off || static_cast<int>(level) < g_level.load()) return;
  const auto elapsed = std::chrono::steady_clock::now() - g_start;
  nlohmann::json line;
  line["ts_ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  line["level"] = level_name(level);
  line["event"] = event;
  for (const auto& [key, value] : fields) line[key] = value;
  const std::string text = line.dump();
  std::lock_guard<std::mutex> lock(g_write_mutex);
  std::fprintf(stderr, "%s\n", text.c_str());
}

}  // namespace eio
