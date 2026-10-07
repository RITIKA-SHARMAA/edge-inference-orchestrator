// Tensor: a float32 array plus its shape, stored row-major (last dimension fastest).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace eio {

struct Tensor {
  std::vector<int64_t> shape;
  std::vector<float> data;

  // Number of elements implied by the shape (1 for a scalar shape {}).
  size_t element_count() const;
};

// Tensors keyed by name, e.g. {"input": image}. std::map keeps output order stable.
using NamedTensors = std::map<std::string, Tensor>;

size_t element_count(const std::vector<int64_t>& shape);
std::string shape_to_string(const std::vector<int64_t>& shape);

// True if `actual` fits `expected`, where -1 in `expected` means "any size".
bool shape_matches(const std::vector<int64_t>& expected, const std::vector<int64_t>& actual);

// True if two declared shapes could describe the same tensor (-1 matches anything).
bool shapes_compatible(const std::vector<int64_t>& a, const std::vector<int64_t>& b);

}  // namespace eio
