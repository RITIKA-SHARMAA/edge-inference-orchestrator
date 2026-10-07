#include <memory>

#include <gtest/gtest.h>

#include "backend.h"
#include "test_helpers.h"

using eio::Backend;
using eio::BackendRegistry;
using eio::Code;

TEST(PluginLoader, LoadsMockAndCpuBackends) {
  std::unique_ptr<Backend> mock, cpu;
  ASSERT_TRUE(Backend::load(eio_test::plugin("eio_backend_mock"), mock).ok());
  ASSERT_TRUE(Backend::load(eio_test::plugin("eio_backend_cpu"), cpu).ok());
  EXPECT_EQ(mock->name(), "mock");
  EXPECT_EQ(cpu->name(), "cpu");
}

TEST(PluginLoader, MissingFileIsNotFound) {
  std::unique_ptr<Backend> backend;
  const eio::Status status = Backend::load(eio_test::plugin("eio_backend_nope"), backend);
  EXPECT_EQ(status.code(), Code::NotFound);
  EXPECT_EQ(backend, nullptr);
}

TEST(PluginLoader, RejectsLibraryWithoutEntrySymbol) {
  std::unique_ptr<Backend> backend;
  const eio::Status status = Backend::load(eio_test::bad_plugin("eio_backend_bad_symbol"), backend);
  EXPECT_EQ(status.code(), Code::PluginError);
  EXPECT_NE(status.message().find("eio_plugin_get_api"), std::string::npos);
}

TEST(PluginLoader, RejectsWrongApiVersion) {
  std::unique_ptr<Backend> backend;
  const eio::Status status =
      Backend::load(eio_test::bad_plugin("eio_backend_bad_version"), backend);
  EXPECT_EQ(status.code(), Code::PluginError);
  EXPECT_NE(status.message().find("version"), std::string::npos);
}

TEST(BackendRegistry, FindsPluginsByName) {
  const BackendRegistry& registry = eio_test::registry();
  EXPECT_NE(registry.find("cpu"), nullptr);
  EXPECT_NE(registry.find("mock"), nullptr);
  EXPECT_EQ(registry.find("gpu"), nullptr);
}

TEST(BackendRegistry, SkipsBrokenPluginsButKeepsTheirErrors) {
  BackendRegistry registry;
  ASSERT_TRUE(registry.load_directory(EIO_TEST_BAD_PLUGIN_DIR).ok());
  EXPECT_TRUE(registry.names().empty());
  EXPECT_EQ(registry.load_errors().size(), 2u);
}

TEST(BackendRegistry, MissingFolderIsNotFound) {
  BackendRegistry registry;
  EXPECT_EQ(registry.load_directory("no/such/folder").code(), Code::NotFound);
}
