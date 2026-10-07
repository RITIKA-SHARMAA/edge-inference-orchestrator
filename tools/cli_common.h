// Helpers shared by eio_run and eio_bench: argument parsing, RAII wrappers for the
// C API handles, .npy loading and top-k printing.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "eio/eio.h"
#include "tensor.h"

namespace cli {

// unique_ptr wrappers so C handles are always destroyed, even on early return.
struct RuntimeDeleter {
  void operator()(eio_runtime* r) const { eio_runtime_destroy(r); }
};
struct SessionDeleter {
  void operator()(eio_session* s) const { eio_session_destroy(s); }
};
struct PipelineDeleter {
  void operator()(eio_pipeline* p) const { eio_pipeline_destroy(p); }
};
struct OutputsDeleter {
  void operator()(eio_outputs* o) const { eio_outputs_destroy(o); }
};
using Runtime = std::unique_ptr<eio_runtime, RuntimeDeleter>;
using Session = std::unique_ptr<eio_session, SessionDeleter>;
using Pipeline = std::unique_ptr<eio_pipeline, PipelineDeleter>;
using Outputs = std::unique_ptr<eio_outputs, OutputsDeleter>;

// EIO_PLUGIN_DIR if set, otherwise "<folder of this executable>/plugins".
std::string default_plugin_dir(const char* argv0);

// Prints "error: <message from eio_last_error()>" and returns 1.
int fail(const std::string& what);

// A named input read from a .npy file; keeps the tensor alive for the eio_input view.
struct NamedInput {
  std::string name;
  eio::Tensor tensor;
  eio_input view() const;
};
bool load_input(const std::string& name, const std::string& path, NamedInput& out,
                std::string& error);

std::vector<std::string> read_lines(const std::string& path);

// Indices of the k largest values, largest first.
std::vector<size_t> top_k(const float* values, size_t count, size_t k);

// Value of "--flag value", or fallback.
std::string arg_value(const std::vector<std::string>& args, const std::string& flag,
                      const std::string& fallback = "");
bool has_flag(const std::vector<std::string>& args, const std::string& flag);
std::vector<std::string> arg_values(const std::vector<std::string>& args, const std::string& flag);

}  // namespace cli
