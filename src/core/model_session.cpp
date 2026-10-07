#include "model_session.h"

#include <set>

#include "log.h"

namespace eio {

namespace {
constexpr size_t kErrLen = 512;
}

Status ModelSession::create(const Backend& backend, const std::string& model_path,
                            std::unique_ptr<ModelSession>& out) {
  std::unique_ptr<ModelSession> session(new ModelSession(backend, model_path));
  char err[kErrLen] = {0};
  if (backend.api().load(model_path.c_str(), &session->model_, err, kErrLen) != 0) {
    session->model_ = nullptr;
    return {Code::BackendError,
            "backend '" + backend.name() + "' could not load " + model_path + ": " + err};
  }
  Status status = session->read_io(false, session->inputs_);
  if (status.ok()) status = session->read_io(true, session->outputs_);
  if (!status.ok()) return status;
  log(LogLevel::Info, "model_loaded", {{"model", model_path}, {"backend", backend.name()}});
  out = std::move(session);
  return Status::Ok();
}

ModelSession::~ModelSession() {
  if (model_) backend_.api().unload(model_);
}

Status ModelSession::read_io(bool is_output, std::vector<IoInfo>& out) {
  const auto& api = backend_.api();
  const size_t count = api.io_count(model_, is_output ? 1 : 0);
  for (size_t i = 0; i < count; ++i) {
    eio_io_desc desc{};
    if (api.io_desc(model_, is_output ? 1 : 0, i, &desc) != 0) {
      return {Code::BackendError, "backend could not describe model " +
                                      std::string(is_output ? "output " : "input ") +
                                      std::to_string(i)};
    }
    // The C API moves float32 tensors only; reject anything else up front.
    if (desc.dtype != EIO_DTYPE_FLOAT32) {
      return {Code::InvalidArgument, model_path_ + ": " + (is_output ? "output '" : "input '") +
                                         desc.name +
                                         "' is not float32, which this SDK does not support"};
    }
    if (desc.rank > EIO_MAX_RANK) {
      return {Code::BackendError,
              model_path_ + ": tensor rank above " + std::to_string(EIO_MAX_RANK)};
    }
    out.push_back({desc.name, std::vector<int64_t>(desc.shape, desc.shape + desc.rank)});
  }
  return Status::Ok();
}

Status ModelSession::validate(const NamedTensors& inputs) const {
  std::set<std::string> expected;
  for (const auto& info : inputs_) {
    expected.insert(info.name);
    auto it = inputs.find(info.name);
    if (it == inputs.end()) {
      return {Code::InvalidArgument, "missing input '" + info.name + "' for " + model_path_};
    }
    const Tensor& tensor = it->second;
    if (!shape_matches(info.shape, tensor.shape)) {
      return {Code::InvalidArgument, "input '" + info.name + "' has shape " +
                                         shape_to_string(tensor.shape) + ", model expects " +
                                         shape_to_string(info.shape)};
    }
    if (tensor.data.size() != tensor.element_count()) {
      return {Code::InvalidArgument,
              "input '" + info.name + "' holds " + std::to_string(tensor.data.size()) +
                  " values but its shape needs " + std::to_string(tensor.element_count())};
    }
  }
  for (const auto& [name, tensor] : inputs) {
    if (!expected.count(name)) {
      return {Code::InvalidArgument, "model " + model_path_ + " has no input named '" + name + "'"};
    }
  }
  return Status::Ok();
}

Status ModelSession::run(const NamedTensors& inputs, NamedTensors& outputs) const {
  Status status = validate(inputs);
  if (!status.ok()) return status;

  // Build the C view of the inputs. Data is not copied: the backend reads our buffers.
  std::vector<eio_plugin_tensor> c_inputs;
  for (const auto& info : inputs_) {
    const Tensor& tensor = inputs.at(info.name);
    eio_plugin_tensor t{};
    t.name = info.name.c_str();
    t.rank = tensor.shape.size();
    for (size_t d = 0; d < t.rank; ++d) t.shape[d] = tensor.shape[d];
    t.data = const_cast<float*>(tensor.data.data());  // backends treat inputs as read-only
    t.count = tensor.data.size();
    c_inputs.push_back(t);
  }

  eio_plugin_tensor* c_outputs = nullptr;
  size_t n_outputs = 0;
  char err[kErrLen] = {0};
  const auto& api = backend_.api();
  if (api.run(model_, c_inputs.data(), c_inputs.size(), &c_outputs, &n_outputs, err, kErrLen) !=
      0) {
    return {Code::BackendError,
            "backend '" + backend_.name() + "' failed running " + model_path_ + ": " + err};
  }

  // Copy results into our own tensors, then hand the plugin's memory back to it.
  // Memory must be freed by the library that allocated it (important on Windows,
  // where each DLL can have its own heap).
  outputs.clear();
  for (size_t i = 0; i < n_outputs; ++i) {
    const eio_plugin_tensor& t = c_outputs[i];
    Tensor tensor;
    tensor.shape.assign(t.shape, t.shape + t.rank);
    tensor.data.assign(t.data, t.data + t.count);
    outputs[t.name] = std::move(tensor);
  }
  api.free_outputs(c_outputs, n_outputs);
  return Status::Ok();
}

}  // namespace eio
