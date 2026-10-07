# Third-party dependencies, all pinned.
#  - ONNX Runtime 1.20.1: prebuilt release archive for the current platform.
#  - nlohmann/json 3.11.3: header-only JSON parser.
#  - GoogleTest 1.15.2: unit tests (only when EIO_BUILD_TESTS is ON).

include(FetchContent)
find_package(Threads REQUIRED)

set(EIO_ORT_VERSION "1.20.1")
set(_ort_base "https://github.com/microsoft/onnxruntime/releases/download/v${EIO_ORT_VERSION}")

if(WIN32)
  set(_ort_pkg "onnxruntime-win-x64-${EIO_ORT_VERSION}.zip")
elseif(APPLE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(_ort_pkg "onnxruntime-osx-arm64-${EIO_ORT_VERSION}.tgz")
  else()
    set(_ort_pkg "onnxruntime-osx-x86_64-${EIO_ORT_VERSION}.tgz")
  endif()
else()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_ort_pkg "onnxruntime-linux-aarch64-${EIO_ORT_VERSION}.tgz")
  else()
    set(_ort_pkg "onnxruntime-linux-x64-${EIO_ORT_VERSION}.tgz")
  endif()
endif()

# Set EIO_ORT_ROOT to an extracted ONNX Runtime package to skip the download.
if(NOT EIO_ORT_ROOT)
  FetchContent_Declare(onnxruntime_prebuilt URL "${_ort_base}/${_ort_pkg}" DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(onnxruntime_prebuilt)
  set(EIO_ORT_ROOT "${onnxruntime_prebuilt_SOURCE_DIR}")
endif()

set(EIO_ORT_LIB_DIR "${EIO_ORT_ROOT}/lib")
add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
target_include_directories(onnxruntime::onnxruntime INTERFACE "${EIO_ORT_ROOT}/include")
if(WIN32)
  set(EIO_ORT_DLL "${EIO_ORT_LIB_DIR}/onnxruntime.dll")
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${EIO_ORT_DLL}"
    IMPORTED_IMPLIB "${EIO_ORT_LIB_DIR}/onnxruntime.lib")
elseif(APPLE)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${EIO_ORT_LIB_DIR}/libonnxruntime.${EIO_ORT_VERSION}.dylib")
else()
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${EIO_ORT_LIB_DIR}/libonnxruntime.so.${EIO_ORT_VERSION}")
endif()

FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(nlohmann_json)

if(EIO_BUILD_TESTS)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # match MSVC's default runtime
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()
