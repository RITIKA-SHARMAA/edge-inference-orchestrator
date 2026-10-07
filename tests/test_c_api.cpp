// Tests that use only include/eio/eio.h, linked against the shared library,
// exactly the way an application or the Python bindings use it.
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "eio/eio.h"

namespace {

struct RuntimeFixture : ::testing::Test {
  eio_runtime* runtime = nullptr;
  void SetUp() override { ASSERT_EQ(eio_runtime_create(EIO_TEST_PLUGIN_DIR, &runtime), EIO_OK); }
  void TearDown() override { eio_runtime_destroy(runtime); }
};

}  // namespace

TEST(CApi, Version) { EXPECT_STREQ(eio_version(), "0.1.0"); }

TEST(CApi, NullArgumentsAreRejectedNotCrashed) {
  EXPECT_EQ(eio_runtime_create(nullptr, nullptr), EIO_INVALID_ARGUMENT);
  EXPECT_NE(std::string(eio_last_error()), "");
  EXPECT_EQ(eio_session_run(nullptr, nullptr, 0, nullptr), EIO_INVALID_ARGUMENT);
  eio_runtime_destroy(nullptr);  // destroy functions accept null, like free()
  eio_outputs_destroy(nullptr);
}

TEST(CApi, MissingPluginFolder) {
  eio_runtime* runtime = nullptr;
  EXPECT_EQ(eio_runtime_create("no/such/dir", &runtime), EIO_NOT_FOUND);
  EXPECT_EQ(runtime, nullptr);
}

TEST_F(RuntimeFixture, ListsBackends) {
  std::vector<std::string> names;
  for (size_t i = 0; i < eio_runtime_backend_count(runtime); ++i) {
    names.push_back(eio_runtime_backend_name(runtime, i));
  }
  EXPECT_EQ(names, (std::vector<std::string>{"cpu", "mock"}));
  EXPECT_EQ(eio_runtime_backend_name(runtime, 99), nullptr);
}

TEST_F(RuntimeFixture, RunsAModelEndToEnd) {
  eio_session* session = nullptr;
  ASSERT_EQ(eio_session_create(runtime, "cpu", EIO_TEST_DATA_DIR "/add_one.onnx", &session),
            EIO_OK);

  const char* name;
  const int64_t* shape;
  size_t rank;
  ASSERT_EQ(eio_session_io_info(session, 0, 0, &name, &shape, &rank), EIO_OK);
  EXPECT_STREQ(name, "x");
  EXPECT_EQ(rank, 2u);

  const int64_t in_shape[] = {1, 4};
  const float in_data[] = {1, 2, 3, 4};
  const eio_input input{"x", in_shape, 2, in_data, 4};
  eio_outputs* outputs = nullptr;
  ASSERT_EQ(eio_session_run(session, &input, 1, &outputs), EIO_OK);
  ASSERT_EQ(eio_outputs_count(outputs), 1u);

  const float* data;
  size_t count;
  ASSERT_EQ(eio_outputs_get(outputs, 0, &name, &shape, &rank, &data, &count), EIO_OK);
  EXPECT_STREQ(name, "y");
  EXPECT_EQ(std::vector<float>(data, data + count), (std::vector<float>{2, 3, 4, 5}));

  eio_outputs_destroy(outputs);
  eio_session_destroy(session);
}

TEST_F(RuntimeFixture, ErrorsCarryAMessage) {
  eio_session* session = nullptr;
  EXPECT_EQ(eio_session_create(runtime, "gpu", "x.onnx", &session), EIO_NOT_FOUND);
  EXPECT_NE(std::string(eio_last_error()).find("gpu"), std::string::npos);

  ASSERT_EQ(eio_session_create(runtime, "cpu", EIO_TEST_DATA_DIR "/add_one.onnx", &session),
            EIO_OK);
  const int64_t bad_shape[] = {1, 5};
  const float zeros[5] = {};
  const eio_input input{"x", bad_shape, 2, zeros, 5};
  eio_outputs* outputs = nullptr;
  EXPECT_EQ(eio_session_run(session, &input, 1, &outputs), EIO_INVALID_ARGUMENT);
  EXPECT_EQ(outputs, nullptr);
  EXPECT_NE(std::string(eio_last_error()).find("[1, 5]"), std::string::npos);
  eio_session_destroy(session);
}

TEST_F(RuntimeFixture, RunsAPipelineAndReportsTimings) {
  eio_pipeline* pipeline = nullptr;
  ASSERT_EQ(eio_pipeline_load(runtime, EIO_TEST_DATA_DIR "/parallel.json", 2, &pipeline), EIO_OK);
  const int64_t shape[] = {1, 4};
  const float data[] = {1, 2, 3, 4};
  const eio_input input{"x", shape, 2, data, 4};
  eio_outputs* outputs = nullptr;
  ASSERT_EQ(eio_pipeline_run(pipeline, &input, 1, &outputs), EIO_OK);
  EXPECT_EQ(eio_pipeline_timing_count(pipeline), 3u);

  const char* node;
  double start, end;
  ASSERT_EQ(eio_pipeline_timing(pipeline, 0, &node, &start, &end), EIO_OK);
  EXPECT_GE(end, start);
  eio_outputs_destroy(outputs);
  eio_pipeline_destroy(pipeline);
}
