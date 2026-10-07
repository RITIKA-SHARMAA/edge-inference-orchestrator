#include "dynamic_library.h"

#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <filesystem>
#else
#include <dlfcn.h>
#endif

namespace eio {

DynamicLibrary::~DynamicLibrary() { close(); }

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)), path_(std::move(other.path_)) {}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept {
  if (this != &other) {
    close();
    handle_ = std::exchange(other.handle_, nullptr);
    path_ = std::move(other.path_);
  }
  return *this;
}

Status DynamicLibrary::open(const std::string& path) {
  close();
#if defined(_WIN32)
  // LOAD_WITH_ALTERED_SEARCH_PATH makes Windows look for the plugin's own dependencies
  // (e.g. onnxruntime.dll) in the plugin's folder instead of only next to the .exe.
  // The flag only applies to absolute paths written with backslashes.
  const std::wstring native =
      std::filesystem::absolute(std::filesystem::path(path)).make_preferred().wstring();
  HMODULE module = LoadLibraryExW(native.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!module) {
    return {Code::NotFound,
            "cannot load " + path + " (Windows error " + std::to_string(GetLastError()) + ")"};
  }
  handle_ = module;
#else
  // RTLD_NOW resolves every symbol immediately, so a broken plugin fails here
  // instead of crashing later in the middle of a run.
  int flags = RTLD_NOW | RTLD_LOCAL;
#if defined(EIO_KEEP_MODULES_LOADED)
  // Sanitizer builds only: keep libraries mapped after dlclose so LeakSanitizer can
  // still name the library a leak came from when it reports at exit.
  flags |= RTLD_NODELETE;
#endif
  handle_ = dlopen(path.c_str(), flags);
  if (!handle_) {
    const char* reason = dlerror();
    return {Code::NotFound, "cannot load " + path + ": " + (reason ? reason : "unknown error")};
  }
#endif
  path_ = path;
  return Status::Ok();
}

void* DynamicLibrary::symbol(const char* name) const {
  if (!handle_) return nullptr;
#if defined(_WIN32)
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
  return dlsym(handle_, name);
#endif
}

void DynamicLibrary::close() {
  if (!handle_) return;
#if defined(_WIN32)
  FreeLibrary(static_cast<HMODULE>(handle_));
#else
  dlclose(handle_);
#endif
  handle_ = nullptr;
}

std::string DynamicLibrary::module_suffix() {
#if defined(_WIN32)
  return ".dll";
#else
  // CMake MODULE libraries use .so on both Linux and macOS.
  return ".so";
#endif
}

}  // namespace eio
