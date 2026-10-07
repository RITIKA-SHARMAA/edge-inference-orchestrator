#include "npy.h"

#include <cstring>
#include <fstream>
#include <regex>

namespace eio {

namespace {

// The header is a Python dict literal, e.g.
// {'descr': '<f4', 'fortran_order': False, 'shape': (1, 3, 224, 224), }
Status parse_header(const std::string& header, std::vector<int64_t>& shape) {
  std::smatch m;
  if (!std::regex_search(header, m, std::regex("'descr':\\s*'([^']+)'"))) {
    return {Code::InvalidArgument, "npy header has no descr"};
  }
  if (m[1] != "<f4") {
    return {Code::InvalidArgument,
            "only little-endian float32 .npy files are supported, got dtype " + m[1].str()};
  }
  if (!std::regex_search(header, m, std::regex("'fortran_order':\\s*(True|False)")) ||
      m[1] != "False") {
    return {Code::InvalidArgument, "Fortran-ordered .npy files are not supported"};
  }
  if (!std::regex_search(header, m, std::regex("'shape':\\s*\\(([^)]*)\\)"))) {
    return {Code::InvalidArgument, "npy header has no shape"};
  }
  shape.clear();
  const std::string dims = m[1].str();
  std::regex number("\\d+");
  for (auto it = std::sregex_iterator(dims.begin(), dims.end(), number);
       it != std::sregex_iterator(); ++it) {
    shape.push_back(std::stoll(it->str()));
  }
  return Status::Ok();
}

}  // namespace

Status read_npy(const std::string& path, Tensor& out) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return {Code::NotFound, "cannot open " + path};

  char magic[6];
  file.read(magic, 6);
  if (!file || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
    return {Code::InvalidArgument, path + " is not a .npy file"};
  }
  unsigned char version[2];
  file.read(reinterpret_cast<char*>(version), 2);

  // v1 stores the header length in 2 bytes, v2/v3 in 4 bytes (little-endian).
  uint32_t header_len = 0;
  if (version[0] == 1) {
    unsigned char b[2];
    file.read(reinterpret_cast<char*>(b), 2);
    header_len = static_cast<uint32_t>(b[0] | (b[1] << 8));
  } else {
    unsigned char b[4];
    file.read(reinterpret_cast<char*>(b), 4);
    header_len = static_cast<uint32_t>(b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24));
  }
  std::string header(header_len, '\0');
  file.read(header.data(), header_len);
  if (!file) return {Code::InvalidArgument, path + ": truncated header"};

  Tensor tensor;
  Status status = parse_header(header, tensor.shape);
  if (!status.ok()) return {status.code(), path + ": " + status.message()};

  tensor.data.resize(tensor.element_count());
  file.read(reinterpret_cast<char*>(tensor.data.data()),
            static_cast<std::streamsize>(tensor.data.size() * sizeof(float)));
  if (!file) return {Code::InvalidArgument, path + ": file is shorter than its shape says"};

  out = std::move(tensor);
  return Status::Ok();
}

Status write_npy(const std::string& path, const Tensor& tensor) {
  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < tensor.shape.size(); ++i) {
    header += std::to_string(tensor.shape[i]);
    header += (tensor.shape.size() == 1 || i + 1 < tensor.shape.size()) ? ", " : "";
  }
  header += "), }";
  // Pad with spaces so magic(6) + version(2) + len(2) + header is a multiple of 64.
  const size_t total = 10 + header.size() + 1;
  header.append((64 - total % 64) % 64, ' ');
  header += '\n';

  std::ofstream file(path, std::ios::binary);
  if (!file) return {Code::NotFound, "cannot write " + path};
  file.write("\x93NUMPY\x01\x00", 8);
  const auto len = static_cast<uint16_t>(header.size());
  const char len_bytes[2] = {static_cast<char>(len & 0xff), static_cast<char>(len >> 8)};
  file.write(len_bytes, 2);
  file.write(header.data(), static_cast<std::streamsize>(header.size()));
  file.write(reinterpret_cast<const char*>(tensor.data.data()),
             static_cast<std::streamsize>(tensor.data.size() * sizeof(float)));
  return file ? Status::Ok() : Status{Code::Internal, "failed writing " + path};
}

}  // namespace eio
