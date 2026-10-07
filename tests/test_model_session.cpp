#include <filesystem>
#include <memory>

#include <gtest/gtest.h>

#include "model_session.h"
#include "npy.h"
#include "test_helpers.h"

using eio::Code;
using eio::ModelSession;
using eio::NamedTensors;
using eio::Tensor;

namespace {
std::unique_ptr<ModelSession> open(const std::string& backend, const std::string& model) {
  std::unique_ptr<ModelSession> session;
  const eio::Status status =
      ModelSession::create(*eio_test::registry().find(backend), model, session);
  EXPECT_TRUE(status.ok()) << status.message();
  return session;
}
}  // namespace

TEST(ModelSession, ReadsInputAndOutputDescriptions) {
  auto session = open("cpu", eio_test::data("add_one.onnx"));
  ASSERT_EQ(session->inputs().size(), 1u);
  EXPECT_EQ(session->inputs()[0].name, "x");
  EXPECT_EQ(session->inputs()[0].shape, (std::vector<int64_t>{-1, 4}));  // dynamic batch
  EXPECT_EQ(session->outputs()[0].name, "y");
}

TEST(ModelSession, RunsAnOnnxModel) {
  auto session = open("cpu", eio_test::data("add_one.onnx"));
  NamedTensors outputs;
  ASSERT_TRUE(session->run({{"x", Tensor{{2, 4}, {0, 1, 2, 3, 4, 5, 6, 7}}}}, outputs).ok());
  EXPECT_EQ(outputs.at("y").shape, (std::vector<int64_t>{2, 4}));
  EXPECT_EQ(outputs.at("y").data, (std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}));
}

TEST(ModelSession, MissingModelFile) {
  std::unique_ptr<ModelSession> session;
  const eio::Status status =
      ModelSession::create(*eio_test::registry().find("cpu"), "missing.onnx", session);
  EXPECT_EQ(status.code(), Code::BackendError);
  EXPECT_NE(status.message().find("missing.onnx"), std::string::npos);
}

TEST(ModelSession, RejectsNonFloatModels) {
  std::unique_ptr<ModelSession> session;
  const eio::Status status = ModelSession::create(*eio_test::registry().find("cpu"),
                                                  eio_test::data("int64_ids.onnx"), session);
  EXPECT_EQ(status.code(), Code::InvalidArgument);
  EXPECT_NE(status.message().find("not float32"), std::string::npos);
}

TEST(ModelSession, RejectsWrongShapeBeforeCallingTheBackend) {
  auto session = open("cpu", eio_test::data("add_one.onnx"));
  NamedTensors outputs;
  const eio::Status status = session->run({{"x", Tensor{{1, 5}, {0, 0, 0, 0, 0}}}}, outputs);
  EXPECT_EQ(status.code(), Code::InvalidArgument);
  EXPECT_NE(status.message().find("[1, 5]"), std::string::npos);
  EXPECT_NE(status.message().find("[-1, 4]"), std::string::npos);
}

TEST(ModelSession, RejectsDataThatDoesNotMatchItsShape) {
  auto session = open("cpu", eio_test::data("add_one.onnx"));
  NamedTensors outputs;
  const eio::Status status = session->run({{"x", Tensor{{1, 4}, {1, 2}}}}, outputs);
  EXPECT_EQ(status.code(), Code::InvalidArgument);
}

TEST(ModelSession, RejectsMissingAndUnknownInputs) {
  auto session = open("cpu", eio_test::data("add_one.onnx"));
  NamedTensors outputs;
  EXPECT_EQ(session->run({}, outputs).code(), Code::InvalidArgument);
  const eio::Status status =
      session->run({{"x", Tensor{{1, 4}, {0, 0, 0, 0}}}, {"extra", Tensor{{1}, {0}}}}, outputs);
  EXPECT_EQ(status.code(), Code::InvalidArgument);
  EXPECT_NE(status.message().find("extra"), std::string::npos);
}

TEST(ModelSession, MockBackendScales) {
  auto session = open("mock", eio_test::data("mock_double.json"));
  NamedTensors outputs;
  ASSERT_TRUE(session->run({{"x", Tensor{{1, 4}, {1, 2, 3, 4}}}}, outputs).ok());
  EXPECT_EQ(outputs.at("y").data, (std::vector<float>{2, 4, 6, 8}));
}

// Uses the real MobileNetV2 model when scripts/get_models.py has been run.
TEST(ModelSession, MobileNetV2ClassifiesTheSampleCat) {
  const std::string model = std::string(EIO_TEST_MODELS_DIR) + "/mobilenetv2.onnx";
  const std::string sample = std::string(EIO_TEST_SAMPLES_DIR) + "/cat.npy";
  if (!std::filesystem::exists(model) || !std::filesystem::exists(sample)) {
    GTEST_SKIP() << "run scripts/get_models.py and scripts/prepare_data.py first";
  }
  auto session = open("cpu", model);
  Tensor image;
  ASSERT_TRUE(eio::read_npy(sample, image).ok());
  NamedTensors outputs;
  ASSERT_TRUE(session->run({{"input", image}}, outputs).ok());
  const auto& logits = outputs.at("output").data;
  ASSERT_EQ(logits.size(), 1000u);
  const auto best = std::max_element(logits.begin(), logits.end()) - logits.begin();
  // ImageNet classes 281-285 are domestic cats (tabby, tiger cat, Persian, Siamese, Egyptian).
  EXPECT_GE(best, 281);
  EXPECT_LE(best, 285);
}
