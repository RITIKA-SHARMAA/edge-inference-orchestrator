#include "builtin_ops.h"

#include <algorithm>
#include <cmath>

namespace eio {

bool is_builtin_op(const std::string& op) { return op == "mean" || op == "softmax_mean"; }

std::vector<std::string> builtin_op_names() { return {"mean", "softmax_mean"}; }

void softmax_last_axis(Tensor& tensor) {
  if (tensor.shape.empty()) return;
  const auto width = static_cast<size_t>(tensor.shape.back());
  if (width == 0) return;
  for (size_t row = 0; row + width <= tensor.data.size(); row += width) {
    float* values = tensor.data.data() + row;
    // Subtract the max first so exp() cannot overflow.
    const float max_value = *std::max_element(values, values + width);
    float sum = 0.0f;
    for (size_t i = 0; i < width; ++i) {
      values[i] = std::exp(values[i] - max_value);
      sum += values[i];
    }
    for (size_t i = 0; i < width; ++i) values[i] /= sum;
  }
}

Status run_builtin(const std::string& op, const NamedTensors& inputs, Tensor& out) {
  if (!is_builtin_op(op)) return {Code::InvalidArgument, "unknown built-in op '" + op + "'"};
  if (inputs.size() < 2) return {Code::InvalidArgument, op + " needs at least two inputs"};

  const Tensor& first = inputs.begin()->second;
  Tensor result;
  result.shape = first.shape;
  result.data.assign(first.data.size(), 0.0f);

  for (const auto& [name, input] : inputs) {
    if (input.shape != first.shape) {
      return {Code::InvalidArgument, op + ": input '" + name + "' has shape " +
                                         shape_to_string(input.shape) + ", expected " +
                                         shape_to_string(first.shape)};
    }
    Tensor term = input;
    if (op == "softmax_mean") softmax_last_axis(term);
    for (size_t i = 0; i < term.data.size(); ++i) result.data[i] += term.data[i];
  }
  const auto n = static_cast<float>(inputs.size());
  for (float& value : result.data) value /= n;
  out = std::move(result);
  return Status::Ok();
}

}  // namespace eio
