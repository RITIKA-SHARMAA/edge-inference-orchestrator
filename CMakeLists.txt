cmake_minimum_required(VERSION 3.20)
project(edge_inference_orchestrator VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(EIO_BUILD_TESTS "Build the GoogleTest suite" ON)
option(EIO_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer (GCC/Clang)" OFF)

# Everything runnable lands in build/bin and plugins in build/bin/plugins.
# The `$<0:>` generator expression stops Visual Studio from adding a Debug/Release
# sub-folder, so the layout is identical on every platform.
set(EIO_BIN_DIR "${CMAKE_BINARY_DIR}/bin")
set(EIO_PLUGIN_DIR "${EIO_BIN_DIR}/plugins")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${EIO_BIN_DIR}$<0:>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${EIO_BIN_DIR}$<0:>")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib$<0:>")

# Symbols are hidden unless marked with EIO_API / EIO_PLUGIN_EXPORT, which matches
# how Windows DLLs behave and keeps the Linux/macOS export list small.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

include(cmake/warnings.cmake)
include(cmake/dependencies.cmake)

if(EIO_SANITIZE AND NOT MSVC)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
  add_link_options(-fsanitize=address,undefined)
  add_compile_definitions(EIO_KEEP_MODULES_LOADED)
endif()

# ---------------------------------------------------------------- core library
#
# eio_internal: the C++ implementation, as a static library. The unit tests link it
#               directly so they can test C++ classes that are not exported.
# eio_core:     the shared library (.so/.dylib/.dll) users link. It only exports the
#               C API from include/eio/eio.h; every C++ symbol stays hidden.

add_library(eio_internal STATIC
  src/core/status.cpp
  src/core/tensor.cpp
  src/core/npy.cpp
  src/core/log.cpp
  src/core/dynamic_library.cpp
  src/core/backend.cpp
  src/core/model_session.cpp
  src/core/builtin_ops.cpp
  src/core/pipeline.cpp
  src/core/thread_pool.cpp
  src/core/scheduler.cpp
)
set_target_properties(eio_internal PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(eio_internal
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
target_link_libraries(eio_internal
  PUBLIC nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS} Threads::Threads)
eio_set_warnings(eio_internal)

add_library(eio_core SHARED src/core/c_api.cpp)
target_include_directories(eio_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(eio_core PRIVATE EIO_BUILDING_LIBRARY)
target_link_libraries(eio_core PRIVATE eio_internal)
eio_set_warnings(eio_core)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Also hide the standard-library template code pulled in from eio_internal, so the
  # library exports exactly the C API (check with: nm -D --defined-only libeio_core.so).
  target_link_options(eio_core PRIVATE -Wl,--exclude-libs,ALL)
endif()

# ---------------------------------------------------------------- backend plugins

function(eio_add_plugin target)
  add_library(${target} MODULE ${ARGN})
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  set_target_properties(${target} PROPERTIES
    PREFIX ""                                  # eio_backend_cpu.so, not libeio_backend_cpu.so
    LIBRARY_OUTPUT_DIRECTORY "${EIO_PLUGIN_DIR}$<0:>"
    RUNTIME_OUTPUT_DIRECTORY "${EIO_PLUGIN_DIR}$<0:>")
  eio_set_warnings(${target})
endfunction()

eio_add_plugin(eio_backend_cpu src/plugins/cpu_backend.cpp)
target_link_libraries(eio_backend_cpu PRIVATE onnxruntime::onnxruntime)
# Find libonnxruntime next to the plugin at runtime (Linux/macOS); on Windows the DLL is copied.
set_target_properties(eio_backend_cpu PROPERTIES
  BUILD_RPATH "${EIO_ORT_LIB_DIR}"
  INSTALL_RPATH "$ORIGIN")
if(WIN32)
  add_custom_command(TARGET eio_backend_cpu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${EIO_ORT_DLL}" "${EIO_PLUGIN_DIR}")
endif()

eio_add_plugin(eio_backend_mock src/plugins/mock_backend.cpp)
target_link_libraries(eio_backend_mock PRIVATE nlohmann_json::nlohmann_json)

# ---------------------------------------------------------------- command-line tools

# The tools use only the public C API, exactly like an application would.
# cli_common adds .npy loading (from eio_internal's npy reader) and printing helpers.
add_library(eio_cli_common STATIC tools/cli_common.cpp src/core/npy.cpp src/core/tensor.cpp src/core/status.cpp)
target_include_directories(eio_cli_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
eio_set_warnings(eio_cli_common)

add_executable(eio_run tools/eio_run.cpp)
target_link_libraries(eio_run PRIVATE eio_core eio_cli_common)
eio_set_warnings(eio_run)

add_executable(eio_bench tools/eio_bench.cpp)
target_link_libraries(eio_bench PRIVATE eio_core eio_cli_common nlohmann_json::nlohmann_json)
eio_set_warnings(eio_bench)

# Tools need the plugins to exist before they are useful.
add_dependencies(eio_run eio_backend_cpu eio_backend_mock)
add_dependencies(eio_bench eio_backend_cpu eio_backend_mock)

# ---------------------------------------------------------------- tests

if(EIO_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
