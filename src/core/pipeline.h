// Pipeline: several models (and built-in steps) wired together as a graph,
// described in JSON:
//
// {
//   "inputs":  { "image": [1, 3, 224, 224] },
//   "nodes": [
//     { "id": "fp32", "model": "models/mobilenetv2.onnx", "backend": "cpu",
//       "inputs": { "input": "image" } },
//     { "id": "int8", "model": "models/mobilenetv2.int8.onnx", "backend": "cpu",
//       "inputs": { "input": "image" } },
//     { "id": "combine", "op": "softmax_mean",
//       "inputs": { "a": "fp32.output", "b": "int8.output" } }
//   ],
//   "outputs": { "probabilities": "combine.output" }
// }
//
// A source is either a pipeline input name ("image") or "<node id>.<output name>".
// Loading validates everything up front: unknown nodes or outputs, model inputs
// left unconnected, shape mismatches along edges, and cycles.
#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "backend.h"
#include "model_session.h"
#include "scheduler.h"
#include "status.h"
#include "tensor.h"
#include "thread_pool.h"

namespace eio {

struct PipelineNode {
  std::string id;
  std::string op;  // empty for model nodes, e.g. "softmax_mean" for built-ins
  std::string model_path;
  std::string backend;
  std::map<std::string, std::string> inputs;                  // node input name -> source
  std::unique_ptr<ModelSession> session;                      // model nodes only
  std::map<std::string, std::vector<int64_t>> output_shapes;  // declared shapes
};

class Pipeline {
 public:
  // Relative model paths are resolved against the JSON file's folder.
  static Status load_file(const std::string& path, const BackendRegistry& registry, size_t threads,
                          std::unique_ptr<Pipeline>& out);
  static Status load_string(const std::string& json_text, const std::string& base_dir,
                            const BackendRegistry& registry, size_t threads,
                            std::unique_ptr<Pipeline>& out);

  Status run(const NamedTensors& inputs, NamedTensors& outputs, RunReport* report = nullptr);

  const std::vector<std::string>& execution_order() const { return order_; }
  const std::map<std::string, std::vector<int64_t>>& input_shapes() const { return input_shapes_; }
  const std::map<std::string, std::string>& output_sources() const { return outputs_; }
  size_t threads() const { return pool_->size(); }

 private:
  Pipeline() = default;
  Status validate();
  Status declared_shape(const std::string& source, std::vector<int64_t>& shape) const;
  Status run_node(const PipelineNode& node, std::map<std::string, NamedTensors>& results,
                  const NamedTensors& pipeline_inputs, std::mutex& results_mutex);

  std::map<std::string, std::vector<int64_t>> input_shapes_;
  std::vector<PipelineNode> nodes_;
  std::map<std::string, size_t> index_;  // node id -> position in nodes_
  std::map<std::string, std::string> outputs_;
  Dependencies deps_;
  std::vector<std::string> order_;
  std::unique_ptr<ThreadPool> pool_;
  std::mutex run_mutex_;  // one run at a time per pipeline
};

// Splits "node.output" into its parts; returns false for a plain pipeline input name.
bool split_source(const std::string& source, std::string& node, std::string& output);

}  // namespace eio
