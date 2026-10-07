#include "cli_common.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

#include "npy.h"

namespace cli {

std::string default_plugin_dir(const char* argv0) {
  if (const char* env = std::getenv("EIO_PLUGIN_DIR")) return env;
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::path exe = fs::absolute(fs::path(argv0), ec);
  return (exe.parent_path() / "plugins").string();
}

int fail(const std::string& what) {
  std::cerr << "error: " << what;
  const std::string detail = eio_last_error();
  if (!detail.empty()) std::cerr << ": " << detail;
  std::cerr << "\n";
  return 1;
}

eio_input NamedInput::view() const {
  return {name.c_str(), tensor.shape.data(), tensor.shape.size(), tensor.data.data(),
          tensor.data.size()};
}

bool load_input(const std::string& name, const std::string& path, NamedInput& out,
                std::string& error) {
  out.name = name;
  eio::Status status = eio::read_npy(path, out.tensor);
  if (!status.ok()) error = status.message();
  return status.ok();
}

std::vector<std::string> read_lines(const std::string& path) {
  std::vector<std::string> lines;
  std::ifstream file(path);
  for (std::string line; std::getline(file, line);) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    lines.push_back(line);
  }
  return lines;
}

std::vector<size_t> top_k(const float* values, size_t count, size_t k) {
  std::vector<size_t> index(count);
  std::iota(index.begin(), index.end(), 0);
  k = std::min(k, count);
  std::partial_sort(index.begin(), index.begin() + static_cast<std::ptrdiff_t>(k), index.end(),
                    [&](size_t a, size_t b) { return values[a] > values[b]; });
  index.resize(k);
  return index;
}

std::string arg_value(const std::vector<std::string>& args, const std::string& flag,
                      const std::string& fallback) {
  for (size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == flag) return args[i + 1];
  }
  return fallback;
}

bool has_flag(const std::vector<std::string>& args, const std::string& flag) {
  return std::find(args.begin(), args.end(), flag) != args.end();
}

std::vector<std::string> arg_values(const std::vector<std::string>& args, const std::string& flag) {
  std::vector<std::string> values;
  for (size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == flag) values.push_back(args[i + 1]);
  }
  return values;
}

}  // namespace cli
