// ModelSession: one model loaded by one backend.
//
// Loading asks the backend for the model's inputs and outputs (names, shapes,
// types). Every run is validated against that description BEFORE the backend sees
// it, so a wrong shape fails with a clear message naming the input instead of a
// deep runtime error.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "backend.h"
#include "status.h"
#include "tensor.h"

namespace eio {

struct IoInfo {
  std::string name;
  std::vector<int64_t> shape;  // -1 = dynamic dimension
};

class ModelSession {
 public:
  static Status create(const Backend& backend, const std::string& model_path,
                       std::unique_ptr<ModelSession>& out);
  ~ModelSession();
  ModelSession(const ModelSession&) = delete;
  ModelSession& operator=(const ModelSession&) = delete;

  const std::vector<IoInfo>& inputs() const { return inputs_; }
  const std::vector<IoInfo>& outputs() const { return outputs_; }
  const std::string& model_path() const { return model_path_; }
  const std::string& backend_name() const { return backend_.name(); }

  Status validate(const NamedTensors& inputs) const;
  Status run(const NamedTensors& inputs, NamedTensors& outputs) const;

 private:
  ModelSession(const Backend& backend, std::string model_path)
      : backend_(backend), model_path_(std::move(model_path)) {}
  Status read_io(bool is_output, std::vector<IoInfo>& out);

  const Backend& backend_;
  std::string model_path_;
  void* model_ = nullptr;  // owned by the backend; released with api().unload
  std::vector<IoInfo> inputs_;
  std::vector<IoInfo> outputs_;
};

}  // namespace eio
