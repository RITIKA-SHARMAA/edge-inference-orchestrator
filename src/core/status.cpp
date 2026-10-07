#include "status.h"

namespace eio {

const char* code_name(Code code) {
  switch (code) {
    case Code::Ok:
      return "ok";
    case Code::InvalidArgument:
      return "invalid_argument";
    case Code::NotFound:
      return "not_found";
    case Code::PluginError:
      return "plugin_error";
    case Code::BackendError:
      return "backend_error";
    case Code::PipelineError:
      return "pipeline_error";
    case Code::Internal:
      return "internal";
  }
  return "unknown";
}

}  // namespace eio
