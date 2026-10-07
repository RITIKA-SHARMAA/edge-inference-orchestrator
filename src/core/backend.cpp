#include "backend.h"

#include <algorithm>
#include <filesystem>

#include "log.h"

namespace fs = std::filesystem;

namespace eio {

Status Backend::load(const std::string& path, std::unique_ptr<Backend>& out) {
  auto backend = std::unique_ptr<Backend>(new Backend());
  Status status = backend->library_.open(path);
  if (!status.ok()) return status;

  auto get_api =
      reinterpret_cast<eio_plugin_get_api_fn>(backend->library_.symbol(EIO_PLUGIN_ENTRY_SYMBOL));
  if (!get_api) {
    return {Code::PluginError, path + " does not export " EIO_PLUGIN_ENTRY_SYMBOL};
  }
  const eio_backend_api* api = get_api();
  if (!api) return {Code::PluginError, path + ": " EIO_PLUGIN_ENTRY_SYMBOL " returned null"};
  if (api->api_version != EIO_PLUGIN_API_VERSION) {
    return {Code::PluginError, path + ": plugin API version " + std::to_string(api->api_version) +
                                   ", expected " + std::to_string(EIO_PLUGIN_API_VERSION)};
  }
  if (!api->name || !api->load || !api->unload || !api->io_count || !api->io_desc || !api->run ||
      !api->free_outputs) {
    return {Code::PluginError, path + ": plugin function table is incomplete"};
  }
  backend->api_ = api;
  backend->name_ = api->name;
  out = std::move(backend);
  return Status::Ok();
}

Status BackendRegistry::add(std::unique_ptr<Backend> backend) {
  if (find(backend->name())) {
    return {Code::PluginError, "backend '" + backend->name() + "' is already registered"};
  }
  log(LogLevel::Info, "backend_loaded", {{"name", backend->name()}, {"path", backend->path()}});
  backends_.push_back(std::move(backend));
  return Status::Ok();
}

Status BackendRegistry::load_directory(const std::string& dir) {
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) return {Code::NotFound, "plugin folder not found: " + dir};

  // Sort so backends load in the same order on every machine.
  std::vector<fs::path> candidates;
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    const std::string file = entry.path().filename().string();
    if (entry.is_regular_file() && file.rfind("eio_backend_", 0) == 0 &&
        entry.path().extension() == DynamicLibrary::module_suffix()) {
      candidates.push_back(entry.path());
    }
  }
  std::sort(candidates.begin(), candidates.end());

  for (const auto& path : candidates) {
    std::unique_ptr<Backend> backend;
    Status status = Backend::load(path.string(), backend);
    if (status.ok()) status = add(std::move(backend));
    if (!status.ok()) {
      log(LogLevel::Warning, "backend_skipped",
          {{"path", path.string()}, {"reason", status.message()}});
      load_errors_.push_back(status.message());
    }
  }
  return Status::Ok();
}

const Backend* BackendRegistry::find(const std::string& name) const {
  for (const auto& backend : backends_) {
    if (backend->name() == name) return backend.get();
  }
  return nullptr;
}

std::vector<std::string> BackendRegistry::names() const {
  std::vector<std::string> result;
  for (const auto& backend : backends_) result.push_back(backend->name());
  return result;
}

}  // namespace eio
