#include <filesystem>

#include <gtest/gtest.h>

#include "npy.h"
#include "tensor.h"

using eio::Tensor;

TEST(Tensor, ElementCountAndShapeText) {
  Tensor t{{2, 3, 4}, {}};
  EXPECT_EQ(t.element_count(), 24u);
  EXPECT_EQ(eio::shape_to_string({1, -1, 224}), "[1, -1, 224]");
  EXPECT_EQ(eio::element_count({1, -1}), 0u);  // unknown dimension
}

TEST(Tensor, ShapeMatchingTreatsMinusOneAsAny) {
  EXPECT_TRUE(eio::shape_matches({-1, 3, 224, 224}, {8, 3, 224, 224}));
  EXPECT_FALSE(eio::shape_matches({-1, 3, 224, 224}, {8, 1, 224, 224}));
  EXPECT_FALSE(eio::shape_matches({1, 4}, {1, 4, 1}));
  EXPECT_TRUE(eio::shapes_compatible({-1, 1000}, {1, -1}));
  EXPECT_FALSE(eio::shapes_compatible({1, 1000}, {1, 10}));
}

TEST(Npy, RoundTrip) {
  const auto path = (std::filesystem::temp_directory_path() / "eio_roundtrip.npy").string();
  Tensor written{{2, 3}, {1, 2, 3, 4, 5, 6.5f}};
  ASSERT_TRUE(eio::write_npy(path, written).ok());
  Tensor read;
  ASSERT_TRUE(eio::read_npy(path, read).ok());
  EXPECT_EQ(read.shape, written.shape);
  EXPECT_EQ(read.data, written.data);
}

TEST(Npy, OneDimensionalRoundTrip) {
  const auto path = (std::filesystem::temp_directory_path() / "eio_1d.npy").string();
  ASSERT_TRUE(eio::write_npy(path, Tensor{{3}, {7, 8, 9}}).ok());
  Tensor read;
  ASSERT_TRUE(eio::read_npy(path, read).ok());
  EXPECT_EQ(read.shape, std::vector<int64_t>{3});
}

TEST(Npy, MissingFileIsNotFound) {
  Tensor t;
  EXPECT_EQ(eio::read_npy("does/not/exist.npy", t).code(), eio::Code::NotFound);
}

TEST(Npy, RejectsNonNpyFiles) {
  Tensor t;
  const eio::Status status = eio::read_npy(EIO_TEST_DATA_DIR "/mock_double.json", t);
  EXPECT_EQ(status.code(), eio::Code::InvalidArgument);
}
