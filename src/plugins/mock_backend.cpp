// Mock backend plugin, used by tests and for trying pipelines without real models.
//
// A "model" is a small JSON file instead of an .onnx file:
//   { "inputs":  [{"name": "x", "shape": [1, 4]}],
//     "outputs": [{"name": "y", "shape": [1, 4]}],
//     "op": "scale", "scale": 2.0, "delay_ms": 50 }
// op "identity" copies the first input, "scale" multiplies it, "sum" adds all
// inputs element-wise. delay_ms makes runs take time so tests can check that
// independent pipeline branches really overlap.
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "eio/eio_plugin.h"

namespace {

struct MockModel {
  std::vector<eio_io_desc> inputs;
  std::vector<eio_io_desc> outputs;
  std::string op = "identity";
  float scale = 1.0f;
  int delay_ms = 0;
};

void set_error(char* err, size_t err_len, const std::string& message) {
  if (!err || err_len == 0) return;
  std::strncpy(err, message.c_str(), err_len - 1);
  err[err_len - 1] = '\0';
}

eio_io_desc parse_desc(const nlohmann::json& item) {
  eio_io_desc desc{};
  const auto name = item.at("name").get<std::string>();
  std::strncpy(desc.name, name.c_str(), EIO_NAME_LEN - 1);
  const auto shape = item.at("shape").get<std::vector<int64_t>>();
  desc.rank = shape.size();
  for (size_t i = 0; i < shape.size() && i < EIO_MAX_RANK; ++i) desc.shape[i] = shape[i];
  desc.dtype = item.value("dtype", std::string("float32")) == "float32" ? EIO_DTYPE_FLOAT32
                                                                        : EIO_DTYPE_INT64;
  return desc;
}

int load(const char* model_path, void** model, char* err, size_t err_len) {
  std::ifstream file(model_path);
  if (!file) {
    set_error(err, err_len, "file not found");
    return 1;
  }
  try {
    const auto doc = nlohmann::json::parse(file);
    auto m = std::make_unique<MockModel>();
    for (const auto& item : doc.at("inputs")) m->inputs.push_back(parse_desc(item));
    for (const auto& item : doc.at("outputs")) m->outputs.push_back(parse_desc(item));
    m->op = doc.value("op", std::string("identity"));
    m->scale = doc.value("scale", 1.0f);
    m->delay_ms = doc.value("delay_ms", 0);
    if (m->op != "identity" && m->op != "scale" && m->op != "sum") {
      set_error(err, err_len, "unknown mock op '" + m->op + "'");
      return 1;
    }
    if (m->inputs.empty() || m->outputs.empty()) {
      set_error(err, err_len, "mock model needs at least one input and one output");
      return 1;
    }
    *model = m.release();
    return 0;
  } catch (const std::exception& e) {
    set_error(err, err_len, std::string("invalid mock model: ") + e.what());
    return 1;
  }
}

void unload(void* model) { delete static_cast<MockModel*>(model); }

size_t io_count(void* model, int is_output) {
  auto* m = static_cast<MockModel*>(model);
  return is_output ? m->outputs.size() : m->inputs.size();
}

int io_desc(void* model, int is_output, size_t index, eio_io_desc* desc) {
  auto* m = static_cast<MockModel*>(model);
  const auto& list = is_output ? m->outputs : m->inputs;
  if (index >= list.size()) return 1;
  *desc = list[index];
  return 0;
}

int run(void* model, const eio_plugin_tensor* inputs, size_t n_inputs, eio_plugin_tensor** outputs,
        size_t* n_outputs, char* err, size_t err_len) {
  auto* m = static_cast<MockModel*>(model);
  if (n_inputs == 0) {
    set_error(err, err_len, "no inputs");
    return 1;
  }
  if (m->delay_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(m->delay_ms));

  const eio_plugin_tensor& first = inputs[0];
  std::vector<float> values(first.data, first.data + first.count);
  if (m->op == "scale") {
    for (float& v : values) v *= m->scale;
  } else if (m->op == "sum") {
    for (size_t i = 1; i < n_inputs; ++i) {
      if (inputs[i].count != first.count) {
        set_error(err, err_len, "sum needs inputs of equal size");
        return 1;
      }
      for (size_t k = 0; k < values.size(); ++k) values[k] += inputs[i].data[k];
    }
  }

  // Every declared output gets the same values and the first input's shape.
  auto* out =
      static_cast<eio_plugin_tensor*>(std::calloc(m->outputs.size(), sizeof(eio_plugin_tensor)));
  if (!out) {
    set_error(err, err_len, "out of memory");
    return 1;
  }
  for (size_t i = 0; i < m->outputs.size(); ++i) {
    out[i].name = m->outputs[i].name;
    out[i].rank = first.rank;
    std::memcpy(out[i].shape, first.shape, sizeof(first.shape));
    out[i].count = values.size();
    out[i].data = static_cast<float*>(std::malloc(values.size() * sizeof(float)));
    if (!values.empty()) std::memcpy(out[i].data, values.data(), values.size() * sizeof(float));
  }
  *outputs = out;
  *n_outputs = m->outputs.size();
  return 0;
}

void free_outputs(eio_plugin_tensor* outputs, size_t n_outputs) {
  if (!outputs) return;
  for (size_t i = 0; i < n_outputs; ++i) std::free(outputs[i].data);
  std::free(outputs);
}

const eio_backend_api kApi = {
    EIO_PLUGIN_API_VERSION, "mock", load, unload, io_count, io_desc, run, free_outputs,
};

}  // namespace

extern "C" EIO_PLUGIN_EXPORT const eio_backend_api* eio_plugin_get_api(void) { return &kApi; }
