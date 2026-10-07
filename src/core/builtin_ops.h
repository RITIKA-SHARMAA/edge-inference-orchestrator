// Built-in pipeline steps that are not models: small, deterministic tensor maths
// used to combine model outputs (for example averaging two classifiers).
#pragma once

#include <string>
#include <vector>

#include "status.h"
#include "tensor.h"

namespace eio {

bool is_builtin_op(const std::string& op);
std::vector<std::string> builtin_op_names();

// Inputs are processed in name order; all must have the same shape.
//   mean:         element-wise average
//   softmax_mean: softmax over the last axis of each input, then element-wise average
Status run_builtin(const std::string& op, const NamedTensors& inputs, Tensor& out);

void softmax_last_axis(Tensor& tensor);

}  // namespace eio
