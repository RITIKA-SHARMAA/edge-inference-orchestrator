// The C API (include/eio/eio.h) implemented on top of the C++ core.
//
// Every exported function follows the same pattern:
//   1. check pointers,
//   2. call the C++ code inside guard(), which catches every exception,
//   3. convert the internal Status to an eio_code and remember the message
//      per thread for eio_last_error().
#include "eio/eio.h"

#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "backend.h"
#include "log.h"
#include "model_session.h"
#include "pipeline.h"

struct eio_runtime {
  eio::BackendRegistry registry;
};

struct eio_session {
  std::unique_ptr<eio::ModelSession> session;
};

struct eio_pipeline {
  std::unique_ptr<eio::Pipeline> pipeline;
  eio::RunReport last_report;
};

struct eio_outputs {
  std::vector<std::string> names;
  std::vector<eio::Tensor> tensors;
};

namespace {

thread_local std::string g_last_error;

eio_code to_code(eio::Code code) {
  switch (code) {
    case eio::Code::Ok:
      return EIO_OK;
    case eio::Code::InvalidArgument:
      return EIO_INVALID_ARGUMENT;
    case eio::Code::NotFound:
      return EIO_NOT_FOUND;
    case eio::Code::PluginError:
      return EIO_PLUGIN_ERROR;
    case eio::Code::BackendError:
      return EIO_BACKEND_ERROR;
    case eio::Code::PipelineError:
      return EIO_PIPELINE_ERROR;
    case eio::Code::Internal:
      break;
  }
  return EIO_INTERNAL_ERROR;
}

eio_code report(const eio::Status& status) {
  g_last_error = status.ok() ? "" : status.message();
  if (!status.ok()) {
    eio::log(eio::LogLevel::Error, "api_error",
             {{"code", eio::code_name(status.code())}, {"message", status.message()}});
  }
  return to_code(status.code());
}

eio_code invalid(const char* message) { return report({eio::Code::InvalidArgument, message}); }

// Runs `fn` and turns any C++ exception into EIO_INTERNAL_ERROR, because an
// exception escaping into C or Python code is undefined behaviour.
template <typename Fn>
eio_code guard(Fn&& fn) {
  try {
    return report(fn());
  } catch (const std::exception& e) {
    return report({eio::Code::Internal, std::string("unexpected error: ") + e.what()});
  } catch (...) {
    return report({eio::Code::Internal, "unexpected non-standard exception"});
  }
}

eio::Status to_tensors(const eio_input* inputs, size_t n, eio::NamedTensors& out) {
  for (size_t i = 0; i < n; ++i) {
    const eio_input& in = inputs[i];
    if (!in.name || (!in.data && in.count > 0) || (!in.shape && in.rank > 0)) {
      return {eio::Code::InvalidArgument, "input " + std::to_string(i) + " has null fields"};
    }
    eio::Tensor tensor;
    tensor.shape.assign(in.shape, in.shape + in.rank);
    tensor.data.assign(in.data, in.data + in.count);
    out[in.name] = std::move(tensor);
  }
  return eio::Status::Ok();
}

eio_outputs* to_outputs(eio::NamedTensors&& tensors) {
  auto result = std::make_unique<eio_outputs>();
  for (auto& [name, tensor] : tensors) {
    result->names.push_back(name);
    result->tensors.push_back(std::move(tensor));
  }
  return result.release();
}

}  // namespace

extern "C" {

EIO_API const char* eio_version(void) { return "0.1.0"; }

EIO_API const char* eio_last_error(void) { return g_last_error.c_str(); }

EIO_API void eio_set_log_level(int level) {
  if (level < 0) level = 0;
  if (level > 4) level = 4;
  eio::set_log_level(static_cast<eio::LogLevel>(level));
}

EIO_API eio_code eio_runtime_create(const char* plugin_dir, eio_runtime** out) {
  if (!plugin_dir || !out) return invalid("plugin_dir and out are required");
  *out = nullptr;
  return guard([&] {
    auto runtime = std::make_unique<eio_runtime>();
    eio::Status status = runtime->registry.load_directory(plugin_dir);
    if (status.ok()) *out = runtime.release();
    return status;
  });
}

EIO_API void eio_runtime_destroy(eio_runtime* runtime) { delete runtime; }

EIO_API size_t eio_runtime_backend_count(const eio_runtime* runtime) {
  return runtime ? runtime->registry.names().size() : 0;
}

EIO_API const char* eio_runtime_backend_name(const eio_runtime* runtime, size_t index) {
  if (!runtime) return nullptr;
  const auto names = runtime->registry.names();
  if (index >= names.size()) return nullptr;
  // Return a pointer owned by the backend, which lives as long as the runtime.
  return runtime->registry.find(names[index])->name().c_str();
}

EIO_API eio_code eio_session_create(eio_runtime* runtime, const char* backend,
                                    const char* model_path, eio_session** out) {
  if (!runtime || !backend || !model_path || !out) return invalid("all arguments are required");
  *out = nullptr;
  return guard([&]() -> eio::Status {
    const eio::Backend* found = runtime->registry.find(backend);
    if (!found)
      return {eio::Code::NotFound, std::string("backend '") + backend + "' is not loaded"};
    auto session = std::make_unique<eio_session>();
    eio::Status status = eio::ModelSession::create(*found, model_path, session->session);
    if (status.ok()) *out = session.release();
    return status;
  });
}

EIO_API void eio_session_destroy(eio_session* session) { delete session; }

EIO_API size_t eio_session_io_count(const eio_session* session, int is_output) {
  if (!session) return 0;
  return is_output ? session->session->outputs().size() : session->session->inputs().size();
}

EIO_API eio_code eio_session_io_info(const eio_session* session, int is_output, size_t index,
                                     const char** name, const int64_t** shape, size_t* rank) {
  if (!session || !name || !shape || !rank) return invalid("all arguments are required");
  const auto& list = is_output ? session->session->outputs() : session->session->inputs();
  if (index >= list.size()) return invalid("index out of range");
  *name = list[index].name.c_str();
  *shape = list[index].shape.data();
  *rank = list[index].shape.size();
  return report(eio::Status::Ok());
}

EIO_API eio_code eio_session_run(eio_session* session, const eio_input* inputs, size_t n_inputs,
                                 eio_outputs** out) {
  if (!session || (!inputs && n_inputs > 0) || !out) return invalid("invalid run arguments");
  *out = nullptr;
  return guard([&] {
    eio::NamedTensors in, result;
    eio::Status status = to_tensors(inputs, n_inputs, in);
    if (status.ok()) status = session->session->run(in, result);
    if (status.ok()) *out = to_outputs(std::move(result));
    return status;
  });
}

EIO_API eio_code eio_pipeline_load(eio_runtime* runtime, const char* json_path, size_t threads,
                                   eio_pipeline** out) {
  if (!runtime || !json_path || !out) return invalid("runtime, json_path and out are required");
  *out = nullptr;
  return guard([&] {
    auto pipeline = std::make_unique<eio_pipeline>();
    eio::Status status =
        eio::Pipeline::load_file(json_path, runtime->registry, threads, pipeline->pipeline);
    if (status.ok()) *out = pipeline.release();
    return status;
  });
}

EIO_API void eio_pipeline_destroy(eio_pipeline* pipeline) { delete pipeline; }

EIO_API eio_code eio_pipeline_run(eio_pipeline* pipeline, const eio_input* inputs, size_t n_inputs,
                                  eio_outputs** out) {
  if (!pipeline || (!inputs && n_inputs > 0) || !out) return invalid("invalid run arguments");
  *out = nullptr;
  return guard([&] {
    eio::NamedTensors in, result;
    eio::Status status = to_tensors(inputs, n_inputs, in);
    eio::RunReport run_report;
    if (status.ok()) status = pipeline->pipeline->run(in, result, &run_report);
    if (status.ok()) {
      pipeline->last_report = std::move(run_report);
      *out = to_outputs(std::move(result));
    }
    return status;
  });
}

EIO_API size_t eio_pipeline_timing_count(const eio_pipeline* pipeline) {
  return pipeline ? pipeline->last_report.nodes.size() : 0;
}

EIO_API eio_code eio_pipeline_timing(const eio_pipeline* pipeline, size_t index, const char** node,
                                     double* start_ms, double* end_ms) {
  if (!pipeline || !node || !start_ms || !end_ms) return invalid("all arguments are required");
  if (index >= pipeline->last_report.nodes.size()) return invalid("index out of range");
  const auto& timing = pipeline->last_report.nodes[index];
  *node = timing.id.c_str();
  *start_ms = timing.start_ms;
  *end_ms = timing.end_ms;
  return report(eio::Status::Ok());
}

EIO_API size_t eio_outputs_count(const eio_outputs* outputs) {
  return outputs ? outputs->tensors.size() : 0;
}

EIO_API eio_code eio_outputs_get(const eio_outputs* outputs, size_t index, const char** name,
                                 const int64_t** shape, size_t* rank, const float** data,
                                 size_t* count) {
  if (!outputs || !name || !shape || !rank || !data || !count) {
    return invalid("all arguments are required");
  }
  if (index >= outputs->tensors.size()) return invalid("index out of range");
  const eio::Tensor& tensor = outputs->tensors[index];
  *name = outputs->names[index].c_str();
  *shape = tensor.shape.data();
  *rank = tensor.shape.size();
  *data = tensor.data.data();
  *count = tensor.data.size();
  return report(eio::Status::Ok());
}

EIO_API void eio_outputs_destroy(eio_outputs* outputs) { delete outputs; }

}  // extern "C"
