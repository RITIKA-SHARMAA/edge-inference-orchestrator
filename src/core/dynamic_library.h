// DynamicLibrary: open a shared library at runtime and look up a function by name.
//
//   Linux/macOS: dlopen / dlsym / dlclose
//   Windows:     LoadLibraryEx / GetProcAddress / FreeLibrary
//
// This is the only file in the project with platform #ifdefs for library loading.
#pragma once

#include <string>

#include "status.h"

namespace eio {

class DynamicLibrary {
 public:
  DynamicLibrary() = default;
  ~DynamicLibrary();
  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;
  DynamicLibrary(DynamicLibrary&& other) noexcept;
  DynamicLibrary& operator=(DynamicLibrary&& other) noexcept;

  Status open(const std::string& path);
  void* symbol(const char* name) const;  // nullptr if not found
  bool is_open() const { return handle_ != nullptr; }
  const std::string& path() const { return path_; }

  // Platform file name for a module, e.g. "eio_backend_cpu" -> "eio_backend_cpu.so".
  static std::string module_suffix();

 private:
  void close();
  void* handle_ = nullptr;
  std::string path_;
};

}  // namespace eio
