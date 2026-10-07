#include <iostream>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "pipeline.h"
#include "test_helpers.h"

using eio::Code;
using eio::NamedTensors;
using eio::Pipeline;
using eio::Tensor;

namespace {

eio::Status load(const std::string& json, std::unique_ptr<Pipeline>& out, size_t threads = 2) {
  eio::Status status =
      Pipeline::load_string(json, EIO_TEST_DATA_DIR, eio_test::registry(), threads, out);
  if (!status.ok()) std::cerr << "  [load] " << status.message() << "\n";  // shown on failure
  return status;
}

// Builds {"inputs": {"x": [1, 4]}, "nodes": [...], "outputs": {...}}.
std::string doc(const std::string& nodes, const std::string& outputs) {
  return R"({"inputs": {"x": [1, 4]}, "nodes": [)" + nodes + R"(], "outputs": )" + outputs + "}";
}

const std::string kDouble = R"("model": "mock_double.json", "backend": "mock")";

}  // namespace

TEST(Pipeline, RunsAChainAndReturnsNamedOutputs) {
  std::unique_ptr<Pipeline> pipeline;
  const auto json = doc(R"({"id": "first",  )" + kDouble + R"(, "inputs": {"x": "x"}},
                           {"id": "second", )" +
                            kDouble + R"(, "inputs": {"x": "first.y"}})",
                        R"({"result": "second.y"})");
  ASSERT_TRUE(load(json, pipeline).ok());
  NamedTensors outputs;
  ASSERT_TRUE(pipeline->run({{"x", Tensor{{1, 4}, {1, 2, 3, 4}}}}, outputs).ok());
  EXPECT_EQ(outputs.at("result").data, (std::vector<float>{4, 8, 12, 16}));
  EXPECT_EQ(pipeline->execution_order(), (std::vector<std::string>{"first", "second"}));
}

TEST(Pipeline, IndependentBranchesRunInParallel) {
  std::unique_ptr<Pipeline> pipeline;
  ASSERT_TRUE(
      Pipeline::load_file(eio_test::data("parallel.json"), eio_test::registry(), 2, pipeline).ok());
  NamedTensors outputs;
  eio::RunReport report;
  ASSERT_TRUE(pipeline->run({{"x", Tensor{{1, 4}, {1, 1, 1, 1}}}}, outputs, &report).ok());
  EXPECT_EQ(outputs.at("result").data, (std::vector<float>{2, 2, 2, 2}));

  const eio::NodeTiming* left = nullptr;
  const eio::NodeTiming* right = nullptr;
  for (const auto& t : report.nodes) {
    if (t.id == "left") left = &t;
    if (t.id == "right") right = &t;
  }
  ASSERT_TRUE(left && right);
  EXPECT_LT(std::max(left->start_ms, right->start_ms), std::min(left->end_ms, right->end_ms));
}

TEST(Pipeline, SameResultWithOneOrManyThreads) {
  std::unique_ptr<Pipeline> one, four;
  ASSERT_TRUE(
      Pipeline::load_file(eio_test::data("parallel.json"), eio_test::registry(), 1, one).ok());
  ASSERT_TRUE(
      Pipeline::load_file(eio_test::data("parallel.json"), eio_test::registry(), 4, four).ok());
  const NamedTensors inputs{{"x", Tensor{{1, 4}, {0.5f, -1, 3, 7}}}};
  NamedTensors a, b;
  ASSERT_TRUE(one->run(inputs, a).ok());
  ASSERT_TRUE(four->run(inputs, b).ok());
  EXPECT_EQ(a.at("result").data, b.at("result").data);
}

TEST(Pipeline, SoftmaxMeanCombinesTwoModels) {
  std::unique_ptr<Pipeline> pipeline;
  const auto json = doc(R"({"id": "a", )" + kDouble + R"(, "inputs": {"x": "x"}},
                           {"id": "b", )" +
                            kDouble + R"(, "inputs": {"x": "x"}},
                           {"id": "avg", "op": "softmax_mean", "inputs": {"p": "a.y", "q": "b.y"}})",
                        R"({"probs": "avg.output"})");
  ASSERT_TRUE(load(json, pipeline).ok());
  NamedTensors outputs;
  ASSERT_TRUE(pipeline->run({{"x", Tensor{{1, 4}, {0, 0, 0, 0}}}}, outputs).ok());
  for (float p : outputs.at("probs").data) EXPECT_NEAR(p, 0.25f, 1e-6f);
}

TEST(Pipeline, DetectsCycles) {
  std::unique_ptr<Pipeline> pipeline;
  const auto json = doc(R"({"id": "a", )" + kDouble + R"(, "inputs": {"x": "b.y"}},
                           {"id": "b", )" +
                            kDouble + R"(, "inputs": {"x": "a.y"}})",
                        R"({"out": "b.y"})");
  const eio::Status status = load(json, pipeline);
  EXPECT_EQ(status.code(), Code::PipelineError);
  EXPECT_NE(status.message().find("cycle"), std::string::npos);
}

TEST(Pipeline, RejectsUnknownSources) {
  std::unique_ptr<Pipeline> pipeline;
  auto status =
      load(doc(R"({"id": "a", )" + kDouble + R"(, "inputs": {"x": "ghost.y"}})", R"({"o": "a.y"})"),
           pipeline);
  EXPECT_NE(status.message().find("unknown node 'ghost'"), std::string::npos);
  status =
      load(doc(R"({"id": "a", )" + kDouble + R"(, "inputs": {"x": "image"}})", R"({"o": "a.y"})"),
           pipeline);
  EXPECT_NE(status.message().find("unknown pipeline input 'image'"), std::string::npos);
  status =
      load(doc(R"({"id": "a", )" + kDouble + R"(, "inputs": {"x": "x"}})", R"({"o": "a.nope"})"),
           pipeline);
  EXPECT_NE(status.message().find("has no output 'nope'"), std::string::npos);
}

TEST(Pipeline, RejectsUnconnectedAndUnknownModelInputs) {
  std::unique_ptr<Pipeline> pipeline;
  auto status =
      load(doc(R"({"id": "a", )" + kDouble + R"(, "inputs": {}})", R"({"o": "a.y"})"), pipeline);
  EXPECT_NE(status.message().find("'x' is not connected"), std::string::npos);
  status = load(
      doc(R"({"id": "a", )" + kDouble + R"(, "inputs": {"x": "x", "z": "x"}})", R"({"o": "a.y"})"),
      pipeline);
  EXPECT_NE(status.message().find("no input named 'z'"), std::string::npos);
}

TEST(Pipeline, RejectsShapeMismatchOnAnEdge) {
  std::unique_ptr<Pipeline> pipeline;
  const auto json =
      doc(R"({"id": "wide", "model": "mock_wide.json", "backend": "mock", "inputs": {"x": "x"}})",
          R"({"o": "wide.y"})");
  const eio::Status status = load(json, pipeline);
  EXPECT_EQ(status.code(), Code::PipelineError);
  EXPECT_NE(status.message().find("[1, 4]"), std::string::npos);
  EXPECT_NE(status.message().find("[1, 8]"), std::string::npos);
}

TEST(Pipeline, RejectsBadDocuments) {
  std::unique_ptr<Pipeline> pipeline;
  EXPECT_EQ(load("{not json", pipeline).code(), Code::PipelineError);
  EXPECT_EQ(load(R"({"nodes": []})", pipeline).code(), Code::PipelineError);
  auto status = load(
      doc(R"({"id": "a", "model": "mock_double.json", "backend": "gpu", "inputs": {"x": "x"}})",
          R"({"o": "a.y"})"),
      pipeline);
  EXPECT_NE(status.message().find("backend 'gpu'"), std::string::npos);
  status = load(doc(R"({"id": "a", )" + kDouble + R"(, "inputs": {"x": "x"}}, {"id": "a", )" +
                        kDouble + R"(, "inputs": {"x": "x"}})",
                    R"({"o": "a.y"})"),
                pipeline);
  EXPECT_NE(status.message().find("duplicate"), std::string::npos);
}

TEST(Pipeline, ValidatesInputsAtRunTime) {
  std::unique_ptr<Pipeline> pipeline;
  ASSERT_TRUE(
      load(doc(R"({"id": "a", )" + kDouble + R"(, "inputs": {"x": "x"}})", R"({"o": "a.y"})"),
           pipeline)
          .ok());
  NamedTensors outputs;
  EXPECT_EQ(pipeline->run({}, outputs).code(), Code::InvalidArgument);
  EXPECT_EQ(pipeline->run({{"x", Tensor{{1, 3}, {1, 2, 3}}}}, outputs).code(),
            Code::InvalidArgument);
}
