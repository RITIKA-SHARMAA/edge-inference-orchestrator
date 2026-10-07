// Status: how errors travel inside the library.
//
// Exceptions must never cross the C API (Python or C callers cannot catch them),
// so internal functions return a Status instead. The C API maps Code to eio_code.
#pragma once

#include <string>
#include <utility>

namespace eio {

enum class Code {
  Ok,
  InvalidArgument,
  NotFound,
  PluginError,
  BackendError,
  PipelineError,
  Internal
};

class Status {
 public:
  Status() = default;
  Status(Code code, std::string message) : code_(code), message_(std::move(message)) {}

  static Status Ok() { return {}; }
  bool ok() const { return code_ == Code::Ok; }
  Code code() const { return code_; }
  const std::string& message() const { return message_; }

 private:
  Code code_ = Code::Ok;
  std::string message_;
};

const char* code_name(Code code);

}  // namespace eio
