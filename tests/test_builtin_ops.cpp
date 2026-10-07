#include <cmath>

#include <gtest/gtest.h>

#include "builtin_ops.h"

using eio::Tensor;

TEST(BuiltinOps, MeanAveragesElementWise) {
  Tensor out;
  ASSERT_TRUE(
      eio::run_builtin("mean", {{"a", Tensor{{1, 2}, {1, 3}}}, {"b", Tensor{{1, 2}, {3, 5}}}}, out)
          .ok());
  EXPECT_EQ(out.data, (std::vector<float>{2, 4}));
}

TEST(BuiltinOps, SoftmaxRowsSumToOne) {
  Tensor t{{2, 3}, {1, 2, 3, 1000, 1000, 1000}};  // large values must not overflow
  eio::softmax_last_axis(t);
  EXPECT_NEAR(t.data[0] + t.data[1] + t.data[2], 1.0f, 1e-6f);
  EXPECT_NEAR(t.data[3], 1.0f / 3.0f, 1e-6f);
  EXPECT_GT(t.data[2], t.data[1]);
}

TEST(BuiltinOps, SoftmaxMeanOfIdenticalInputsIsTheirSoftmax) {
  Tensor a{{1, 3}, {0, 1, 2}};
  Tensor out;
  ASSERT_TRUE(eio::run_builtin("softmax_mean", {{"a", a}, {"b", a}}, out).ok());
  eio::softmax_last_axis(a);
  for (size_t i = 0; i < 3; ++i) EXPECT_NEAR(out.data[i], a.data[i], 1e-6f);
}

TEST(BuiltinOps, RejectsMismatchedShapesAndUnknownOps) {
  Tensor out;
  EXPECT_FALSE(eio::run_builtin(
                   "mean", {{"a", Tensor{{1, 2}, {1, 2}}}, {"b", Tensor{{1, 3}, {1, 2, 3}}}}, out)
                   .ok());
  EXPECT_FALSE(
      eio::run_builtin("median", {{"a", Tensor{{1}, {1}}}, {"b", Tensor{{1}, {1}}}}, out).ok());
  EXPECT_FALSE(eio::run_builtin("mean", {{"a", Tensor{{1}, {1}}}}, out).ok());
}
