#include "tensor.h"

#include <sstream>

namespace eio {

size_t element_count(const std::vector<int64_t>& shape) {
  size_t count = 1;
  for (int64_t dim : shape) {
    if (dim < 0) return 0;  // unknown dimension: no concrete size
    count *= static_cast<size_t>(dim);
  }
  return count;
}

size_t Tensor::element_count() const { return eio::element_count(shape); }

std::string shape_to_string(const std::vector<int64_t>& shape) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) out << ", ";
    out << shape[i];
  }
  out << ']';
  return out.str();
}

bool shape_matches(const std::vector<int64_t>& expected, const std::vector<int64_t>& actual) {
  if (expected.size() != actual.size()) return false;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] >= 0 && expected[i] != actual[i]) return false;
  }
  return true;
}

bool shapes_compatible(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] >= 0 && b[i] >= 0 && a[i] != b[i]) return false;
  }
  return true;
}

}  // namespace eio
