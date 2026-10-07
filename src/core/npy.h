// Minimal reader/writer for NumPy .npy files (format version 1.0/2.0, float32,
// little-endian, C order). Python scripts prepare inputs as .npy, and the C++
// tools read them here, so no image-decoding library is needed in C++.
#pragma once

#include <string>

#include "status.h"
#include "tensor.h"

namespace eio {

Status read_npy(const std::string& path, Tensor& out);
Status write_npy(const std::string& path, const Tensor& tensor);

}  // namespace eio
