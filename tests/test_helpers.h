// Shared helpers for the C++ unit tests.
#pragma once

#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "backend.h"
#include "dynamic_library.h"

namespace eio_test {

inline std::string data(const std::string& file) {
  return std::string(EIO_TEST_DATA_DIR) + "/" + file;
}

inline std::string plugin(const std::string& name) {
  return std::string(EIO_TEST_PLUGIN_DIR) + "/" + name + eio::DynamicLibrary::module_suffix();
}

inline std::string bad_plugin(const std::string& name) {
  return std::string(EIO_TEST_BAD_PLUGIN_DIR) + "/" + name + eio::DynamicLibrary::module_suffix();
}

// A registry with the real plugins (cpu + mock), loaded once per test binary.
inline const eio::BackendRegistry& registry() {
  static eio::BackendRegistry instance = [] {
    eio::BackendRegistry r;
    eio::Status status = r.load_directory(EIO_TEST_PLUGIN_DIR);
    if (!status.ok()) ADD_FAILURE() << status.message();
    return r;
  }();
  return instance;
}

}  // namespace eio_test
