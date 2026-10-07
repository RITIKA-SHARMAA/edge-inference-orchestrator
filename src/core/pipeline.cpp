#include "pipeline.h"

#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

#include <nlohmann/json.hpp>

#include "builtin_ops.h"
#include "log.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace eio {

bool split_source(const std::string& source, std::string& node, std::string& output) {
  const auto dot = source.find('.');
  if (dot == std::string::npos) return false;
  node = source.substr(0, dot);
  output = source.substr(dot + 1);
  return true;
}

namespace {
Status fail(const std::string& message) { return {Code::PipelineError, message}; }
}  // namespace

Status Pipeline::load_file(const std::string& path, const BackendRegistry& registry, size_t threads,
                           std::unique_ptr<Pipeline>& out) {
  std::ifstream file(path);
  if (!file) return {Code::NotFound, "pipeline file not found: " + path};
  std::stringstream text;
  text << file.rdbuf();
  const std::string base = fs::path(path).parent_path().string();
  return load_string(text.str(), base, registry, threads, out);
}

Status Pipeline::load_string(const std::string& json_text, const std::string& base_dir,
                             const BackendRegistry& registry, size_t threads,
                             std::unique_ptr<Pipeline>& out) {
  json doc;
  try {
    doc = json::parse(json_text);
  } catch (const json::parse_error& e) {
    return fail(std::string("invalid JSON: ") + e.what());
  }

  std::unique_ptr<Pipeline> pipeline(new Pipeline());
  try {
    if (!doc.contains("nodes") || !doc["nodes"].is_array() || doc["nodes"].empty()) {
      return fail("pipeline needs a non-empty \"nodes\" array");
    }
    // Keep the object in a named variable: iterating .items() of a temporary would dangle.
    const json declared_inputs = doc.value("inputs", json::object());
    for (const auto& [name, shape] : declared_inputs.items()) {
      if (name.find('.') != std::string::npos) {
        return fail("pipeline input names cannot contain '.': " + name);
      }
      pipeline->input_shapes_[name] = shape.get<std::vector<int64_t>>();
    }

    for (const auto& item : doc["nodes"]) {
      PipelineNode node;
      node.id = item.at("id").get<std::string>();
      if (node.id.empty() || node.id.find('.') != std::string::npos) {
        return fail("node ids must be non-empty and cannot contain '.': '" + node.id + "'");
      }
      if (pipeline->index_.count(node.id) || pipeline->input_shapes_.count(node.id)) {
        return fail("duplicate node id '" + node.id + "'");
      }
      node.inputs = item.value("inputs", json::object()).get<std::map<std::string, std::string>>();

      if (item.contains("op")) {
        node.op = item["op"].get<std::string>();
        if (!is_builtin_op(node.op))
          return fail("node '" + node.id + "': unknown op '" + node.op + "'");
      } else {
        node.model_path = item.at("model").get<std::string>();
        node.backend = item.value("backend", std::string("cpu"));
        fs::path model(node.model_path);
        if (model.is_relative() && !base_dir.empty()) model = fs::path(base_dir) / model;
        const Backend* backend = registry.find(node.backend);
        if (!backend)
          return fail("node '" + node.id + "': backend '" + node.backend + "' is not loaded");
        Status status = ModelSession::create(*backend, model.string(), node.session);
        if (!status.ok()) return {status.code(), "node '" + node.id + "': " + status.message()};
      }
      pipeline->index_[node.id] = pipeline->nodes_.size();
      pipeline->nodes_.push_back(std::move(node));
    }
    pipeline->outputs_ = doc.at("outputs").get<std::map<std::string, std::string>>();
  } catch (const json::exception& e) {
    return fail(std::string("malformed pipeline: ") + e.what());
  }

  Status status = pipeline->validate();
  if (!status.ok()) return status;
  pipeline->pool_ = std::make_unique<ThreadPool>(threads);
  log(LogLevel::Info, "pipeline_loaded",
      {{"nodes", std::to_string(pipeline->nodes_.size())}, {"threads", std::to_string(threads)}});
  out = std::move(pipeline);
  return Status::Ok();
}

// The declared (possibly dynamic) shape of whatever `source` refers to.
Status Pipeline::declared_shape(const std::string& source, std::vector<int64_t>& shape) const {
  std::string node_id, output;
  if (!split_source(source, node_id, output)) {
    auto it = input_shapes_.find(source);
    if (it == input_shapes_.end()) return fail("unknown pipeline input '" + source + "'");
    shape = it->second;
    return Status::Ok();
  }
  auto it = index_.find(node_id);
  if (it == index_.end()) return fail("unknown node '" + node_id + "' in '" + source + "'");
  const auto& shapes = nodes_[it->second].output_shapes;
  auto out = shapes.find(output);
  if (out == shapes.end()) return fail("node '" + node_id + "' has no output '" + output + "'");
  shape = out->second;
  return Status::Ok();
}

Status Pipeline::validate() {
  // 1. Dependencies between nodes, and a valid order (this also detects cycles).
  std::vector<std::string> ids;
  for (const auto& node : nodes_) {
    ids.push_back(node.id);
    std::set<std::string> needs;
    for (const auto& [input, source] : node.inputs) {
      std::string dep, output;
      if (split_source(source, dep, output)) {
        if (!index_.count(dep)) {
          return fail("node '" + node.id + "' input '" + input + "' uses unknown node '" + dep +
                      "'");
        }
        if (dep == node.id) return fail("node '" + node.id + "' uses its own output");
        needs.insert(dep);
      } else if (!input_shapes_.count(source)) {
        return fail("node '" + node.id + "' input '" + input + "' uses unknown pipeline input '" +
                    source + "'");
      }
    }
    deps_[node.id] = std::vector<std::string>(needs.begin(), needs.end());
  }
  Status status = order_nodes(ids, deps_, order_);
  if (!status.ok()) return status;

  // 2. Walk in execution order so every source's shape is known before it is used.
  for (const auto& id : order_) {
    PipelineNode& node = nodes_[index_.at(id)];
    if (node.session) {
      for (const auto& info : node.session->inputs()) {
        auto it = node.inputs.find(info.name);
        if (it == node.inputs.end()) {
          return fail("node '" + id + "': model input '" + info.name + "' is not connected");
        }
        std::vector<int64_t> source_shape;
        status = declared_shape(it->second, source_shape);
        if (!status.ok()) return status;
        if (!shapes_compatible(source_shape, info.shape)) {
          return fail("node '" + id + "': '" + it->second + "' has shape " +
                      shape_to_string(source_shape) + " but model input '" + info.name +
                      "' expects " + shape_to_string(info.shape));
        }
      }
      for (const auto& [input, source] : node.inputs) {
        bool known = false;
        for (const auto& info : node.session->inputs()) known = known || info.name == input;
        if (!known) return fail("node '" + id + "': model has no input named '" + input + "'");
      }
      for (const auto& info : node.session->outputs()) node.output_shapes[info.name] = info.shape;
    } else {
      if (node.inputs.size() < 2)
        return fail("node '" + id + "': " + node.op + " needs two or more inputs");
      std::vector<int64_t> first;
      for (const auto& [input, source] : node.inputs) {
        std::vector<int64_t> shape;
        status = declared_shape(source, shape);
        if (!status.ok()) return status;
        if (first.empty()) {
          first = shape;
        } else if (!shapes_compatible(first, shape)) {
          return fail("node '" + id + "': input '" + input + "' has shape " +
                      shape_to_string(shape) + ", which does not match " + shape_to_string(first));
        }
      }
      node.output_shapes["output"] = first;
    }
  }

  // 3. Pipeline outputs must point at real node outputs.
  if (outputs_.empty()) return fail("pipeline declares no outputs");
  for (const auto& [name, source] : outputs_) {
    std::vector<int64_t> shape;
    std::string node_id, output;
    if (!split_source(source, node_id, output)) {
      return fail("output '" + name + "' must reference a node output like 'node.output'");
    }
    status = declared_shape(source, shape);
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

Status Pipeline::run_node(const PipelineNode& node, std::map<std::string, NamedTensors>& results,
                          const NamedTensors& pipeline_inputs, std::mutex& results_mutex) {
  // Gather this node's inputs. Copies are taken under the lock; the model itself runs without it.
  NamedTensors inputs;
  {
    std::lock_guard<std::mutex> lock(results_mutex);
    for (const auto& [input, source] : node.inputs) {
      std::string dep, output;
      if (split_source(source, dep, output)) {
        inputs[input] = results.at(dep).at(output);
      } else {
        inputs[input] = pipeline_inputs.at(source);
      }
    }
  }

  NamedTensors outputs;
  if (node.session) {
    Status status = node.session->run(inputs, outputs);
    if (!status.ok()) return {status.code(), "node '" + node.id + "': " + status.message()};
  } else {
    Tensor result;
    Status status = run_builtin(node.op, inputs, result);
    if (!status.ok()) return {status.code(), "node '" + node.id + "': " + status.message()};
    outputs["output"] = std::move(result);
  }

  std::lock_guard<std::mutex> lock(results_mutex);
  results[node.id] = std::move(outputs);
  return Status::Ok();
}

Status Pipeline::run(const NamedTensors& inputs, NamedTensors& outputs, RunReport* report) {
  std::lock_guard<std::mutex> run_lock(run_mutex_);
  for (const auto& [name, shape] : input_shapes_) {
    auto it = inputs.find(name);
    if (it == inputs.end()) return {Code::InvalidArgument, "missing pipeline input '" + name + "'"};
    if (!shape_matches(shape, it->second.shape)) {
      return {Code::InvalidArgument, "pipeline input '" + name + "' has shape " +
                                         shape_to_string(it->second.shape) + ", expected " +
                                         shape_to_string(shape)};
    }
  }

  std::map<std::string, NamedTensors> results;
  std::mutex results_mutex;
  auto work = [&](const std::string& id) {
    const PipelineNode& node = nodes_[index_.at(id)];
    return run_node(node, results, inputs, results_mutex);
  };
  Status status = run_graph(order_, deps_, *pool_, work, report);
  if (!status.ok()) return status;

  outputs.clear();
  for (const auto& [name, source] : outputs_) {
    std::string node_id, output;
    split_source(source, node_id, output);
    auto node_it = results.find(node_id);
    if (node_it == results.end() || !node_it->second.count(output)) {
      return {Code::BackendError, "node '" + node_id + "' did not produce '" + output + "'"};
    }
    outputs[name] = node_it->second.at(output);
  }
  if (report) {
    for (const auto& timing : report->nodes) {
      log(LogLevel::Debug, "node_timing",
          {{"node", timing.id},
           {"start_ms", std::to_string(timing.start_ms)},
           {"end_ms", std::to_string(timing.end_ms)}});
    }
  }
  return Status::Ok();
}

}  // namespace eio
