include(GoogleTest)

# Two deliberately broken plugins, kept out of the real plugin folder, to test that
# the loader rejects them with a clear error instead of crashing.
foreach(bad bad_symbol bad_version)
  add_library(eio_backend_${bad} MODULE plugins/${bad}.cpp)
  target_include_directories(eio_backend_${bad} PRIVATE ${PROJECT_SOURCE_DIR}/include)
  set_target_properties(eio_backend_${bad} PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${EIO_BIN_DIR}/bad_plugins$<0:>"
    RUNTIME_OUTPUT_DIRECTORY "${EIO_BIN_DIR}/bad_plugins$<0:>")
endforeach()

set(EIO_TEST_DEFINES
  EIO_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
  EIO_TEST_PLUGIN_DIR="${EIO_PLUGIN_DIR}"
  EIO_TEST_BAD_PLUGIN_DIR="${EIO_BIN_DIR}/bad_plugins"
  EIO_TEST_MODELS_DIR="${PROJECT_SOURCE_DIR}/models"
  EIO_TEST_SAMPLES_DIR="${PROJECT_SOURCE_DIR}/samples")

# C++ unit tests against the internal classes.
add_executable(eio_unit_tests
  test_tensor_npy.cpp
  test_plugins.cpp
  test_model_session.cpp
  test_builtin_ops.cpp
  test_scheduler.cpp
  test_pipeline.cpp)
target_link_libraries(eio_unit_tests PRIVATE eio_internal GTest::gtest_main)
target_compile_definitions(eio_unit_tests PRIVATE ${EIO_TEST_DEFINES})
add_dependencies(eio_unit_tests eio_backend_cpu eio_backend_mock eio_backend_bad_symbol eio_backend_bad_version)

# The same library seen from outside, through the exported C API only.
add_executable(eio_c_api_tests test_c_api.cpp)
target_link_libraries(eio_c_api_tests PRIVATE eio_core GTest::gtest_main)
target_compile_definitions(eio_c_api_tests PRIVATE ${EIO_TEST_DEFINES})
add_dependencies(eio_c_api_tests eio_backend_cpu eio_backend_mock)

gtest_discover_tests(eio_unit_tests DISCOVERY_MODE PRE_TEST WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
gtest_discover_tests(eio_c_api_tests DISCOVERY_MODE PRE_TEST WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
