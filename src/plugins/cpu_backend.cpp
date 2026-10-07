// CPU backend plugin: runs ONNX models with ONNX Runtime.
//
// Built as eio_backend_cpu.(so|dll) and loaded at runtime by eio_core. ONNX Runtime
// exceptions are caught here and turned into error codes plus a message, because
// nothing may throw across the plugin's C interface.
#include <onnxruntime_cxx_api.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "eio/eio_plugin.h"

namespace {

void set_error(char* err, size_t err_len, const std::string& message) {
  if (!err || err_len == 0) return;
  std::strncpy(err, message.c_str(), err_len - 1);
  err[err_len - 1] = '\0';
}

// One ONNX Runtime environment per process, created on first use.
Ort::Env& env() {
  static Ort::Env instance(ORT_LOGGING_LEVEL_WARNING, "eio");
  return instance;
}

// Threads ONNX Runtime may use inside ONE model run. Default 1, so parallelism comes
// from the pipeline scheduler and benchmark numbers are stable. EIO_ORT_THREADS overrides.
int intra_op_threads() {
  if (const char* value = std::getenv("EIO_ORT_THREADS")) {
    const int n = std::atoi(value);
    if (n >= 0) return n;  // 0 = let ONNX Runtime decide
  }
  return 1;
}

struct CpuModel {
  std::unique_ptr<Ort::Session> session;
  std::vector<eio_io_desc> inputs;
  std::vector<eio_io_desc> outputs;
  std::vector<std::string> output_names;
};

eio_dtype to_dtype(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return EIO_DTYPE_FLOAT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return EIO_DTYPE_INT64;
    default:
      return EIO_DTYPE_UNSUPPORTED;
  }
}

eio_io_desc describe(const std::string& name, const Ort::TypeInfo& type_info) {
  eio_io_desc desc{};
  std::strncpy(desc.name, name.c_str(), EIO_NAME_LEN - 1);
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    desc.dtype = EIO_DTYPE_UNSUPPORTED;  // sequences, maps, ...
    return desc;
  }
  auto info = type_info.GetTensorTypeAndShapeInfo();
  desc.dtype = to_dtype(info.GetElementType());
  const auto shape = info.GetShape();  // dynamic dimensions come back as -1
  desc.rank = shape.size() > EIO_MAX_RANK ? EIO_MAX_RANK + 1 : shape.size();
  for (size_t i = 0; i < shape.size() && i < EIO_MAX_RANK; ++i) desc.shape[i] = shape[i];
  return desc;
}

int load(const char* model_path, void** model, char* err, size_t err_len) {
  try {
    if (!std::filesystem::exists(model_path)) {
      set_error(err, err_len, "file not found");
      return 1;
    }
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(intra_op_threads());
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    auto result = std::make_unique<CpuModel>();
    // ONNX Runtime takes wide-character paths on Windows.
    const std::filesystem::path path(model_path);
    result->session = std::make_unique<Ort::Session>(env(), path.native().c_str(), options);

    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < result->session->GetInputCount(); ++i) {
      auto name = result->session->GetInputNameAllocated(i, allocator);
      result->inputs.push_back(describe(name.get(), result->session->GetInputTypeInfo(i)));
    }
    for (size_t i = 0; i < result->session->GetOutputCount(); ++i) {
      auto name = result->session->GetOutputNameAllocated(i, allocator);
      result->output_names.emplace_back(name.get());
      result->outputs.push_back(describe(name.get(), result->session->GetOutputTypeInfo(i)));
    }
    *model = result.release();
    return 0;
  } catch (const Ort::Exception& e) {
    set_error(err, err_len, e.what());
  } catch (const std::exception& e) {
    set_error(err, err_len, e.what());
  }
  return 1;
}

void unload(void* model) { delete static_cast<CpuModel*>(model); }

size_t io_count(void* model, int is_output) {
  auto* m = static_cast<CpuModel*>(model);
  return is_output ? m->outputs.size() : m->inputs.size();
}

int io_desc(void* model, int is_output, size_t index, eio_io_desc* desc) {
  auto* m = static_cast<CpuModel*>(model);
  const auto& list = is_output ? m->outputs : m->inputs;
  if (index >= list.size()) return 1;
  *desc = list[index];
  return 0;
}

int run(void* model, const eio_plugin_tensor* inputs, size_t n_inputs, eio_plugin_tensor** outputs,
        size_t* n_outputs, char* err, size_t err_len) {
  auto* m = static_cast<CpuModel*>(model);
  try {
    // Wrap the caller's buffers as ONNX Runtime tensors without copying them.
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> values;
    std::vector<const char*> input_names;
    for (size_t i = 0; i < n_inputs; ++i) {
      values.push_back(Ort::Value::CreateTensor<float>(memory, inputs[i].data, inputs[i].count,
                                                       inputs[i].shape, inputs[i].rank));
      input_names.push_back(inputs[i].name);
    }
    std::vector<const char*> output_names;
    for (const auto& name : m->output_names) output_names.push_back(name.c_str());

    auto results = m->session->Run(Ort::RunOptions{nullptr}, input_names.data(), values.data(),
                                   values.size(), output_names.data(), output_names.size());

    // Copy results into plain malloc'd memory that free_outputs() releases.
    auto* out =
        static_cast<eio_plugin_tensor*>(std::calloc(results.size(), sizeof(eio_plugin_tensor)));
    if (!out) throw std::bad_alloc();
    for (size_t i = 0; i < results.size(); ++i) {
      auto info = results[i].GetTensorTypeAndShapeInfo();
      const auto shape = info.GetShape();
      const size_t count = info.GetElementCount();
      out[i].name = m->output_names[i].c_str();  // owned by CpuModel, outlives the outputs
      out[i].rank = shape.size();
      for (size_t d = 0; d < shape.size() && d < EIO_MAX_RANK; ++d) out[i].shape[d] = shape[d];
      out[i].count = count;
      out[i].data = static_cast<float*>(std::malloc(count * sizeof(float)));
      if (count && !out[i].data) throw std::bad_alloc();
      std::memcpy(out[i].data, results[i].GetTensorData<float>(), count * sizeof(float));
    }
    *outputs = out;
    *n_outputs = results.size();
    return 0;
  } catch (const Ort::Exception& e) {
    set_error(err, err_len, e.what());
  } catch (const std::exception& e) {
    set_error(err, err_len, e.what());
  }
  return 1;
}

void free_outputs(eio_plugin_tensor* outputs, size_t n_outputs) {
  if (!outputs) return;
  for (size_t i = 0; i < n_outputs; ++i) std::free(outputs[i].data);
  std::free(outputs);
}

const eio_backend_api kApi = {
    EIO_PLUGIN_API_VERSION, "cpu", load, unload, io_count, io_desc, run, free_outputs,
};

}  // namespace

extern "C" EIO_PLUGIN_EXPORT const eio_backend_api* eio_plugin_get_api(void) { return &kApi; }
