// Backend: a loaded plugin library plus its validated function table.
// BackendRegistry: every backend found in a plugin folder, looked up by name.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "dynamic_library.h"
#include "eio/eio_plugin.h"
#include "status.h"

namespace eio {

class Backend {
 public:
  // Loads the library, finds eio_plugin_get_api, and checks the table it returns.
  static Status load(const std::string& path, std::unique_ptr<Backend>& out);

  const std::string& name() const { return name_; }
  const eio_backend_api& api() const { return *api_; }
  const std::string& path() const { return library_.path(); }

 private:
  DynamicLibrary library_;
  const eio_backend_api* api_ = nullptr;
  std::string name_;
};

class BackendRegistry {
 public:
  // Loads every file named eio_backend_*<suffix> in `dir`. Files that fail to load
  // are skipped with a warning; their errors are kept in load_errors().
  Status load_directory(const std::string& dir);
  Status add(std::unique_ptr<Backend> backend);

  const Backend* find(const std::string& name) const;
  std::vector<std::string> names() const;
  const std::vector<std::string>& load_errors() const { return load_errors_; }

 private:
  std::vector<std::unique_ptr<Backend>> backends_;
  std::vector<std::string> load_errors_;
};

}  // namespace eio
