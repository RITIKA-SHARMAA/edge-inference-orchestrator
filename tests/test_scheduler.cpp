#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

#include <gtest/gtest.h>

#include "scheduler.h"
#include "thread_pool.h"

using eio::Dependencies;

namespace {
size_t position(const std::vector<std::string>& order, const std::string& id) {
  return static_cast<size_t>(std::find(order.begin(), order.end(), id) - order.begin());
}
}  // namespace

TEST(Kahn, OrdersDependenciesFirst) {
  // a -> b -> d, a -> c -> d
  const Dependencies deps{{"b", {"a"}}, {"c", {"a"}}, {"d", {"b", "c"}}};
  std::vector<std::string> order;
  ASSERT_TRUE(eio::order_nodes({"d", "c", "b", "a"}, deps, order).ok());
  ASSERT_EQ(order.size(), 4u);
  EXPECT_LT(position(order, "a"), position(order, "b"));
  EXPECT_LT(position(order, "a"), position(order, "c"));
  EXPECT_LT(position(order, "b"), position(order, "d"));
  EXPECT_LT(position(order, "c"), position(order, "d"));
}

TEST(Kahn, DetectsCycles) {
  const Dependencies deps{{"a", {"c"}}, {"b", {"a"}}, {"c", {"b"}}, {"d", {}}};
  std::vector<std::string> order;
  const eio::Status status = eio::order_nodes({"a", "b", "c", "d"}, deps, order);
  EXPECT_EQ(status.code(), eio::Code::PipelineError);
  EXPECT_NE(status.message().find("cycle"), std::string::npos);
  EXPECT_NE(status.message().find("a, b, c"), std::string::npos);
  EXPECT_EQ(status.message().find("d"), std::string::npos);  // d is not part of the cycle
}

TEST(Kahn, RejectsUnknownDependencies) {
  std::vector<std::string> order;
  EXPECT_FALSE(eio::order_nodes({"a"}, {{"a", {"ghost"}}}, order).ok());
}

TEST(ThreadPool, RunsEveryTask) {
  std::atomic<int> count{0};
  {
    eio::ThreadPool pool(4);
    for (int i = 0; i < 100; ++i) pool.submit([&] { ++count; });
  }  // destructor waits for the queue to drain
  EXPECT_EQ(count.load(), 100);
}

TEST(RunGraph, RespectsDependencies) {
  const Dependencies deps{{"b", {"a"}}, {"c", {"b"}}};
  eio::ThreadPool pool(3);
  std::mutex mutex;
  std::vector<std::string> ran;
  auto work = [&](const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex);
    ran.push_back(id);
    return eio::Status::Ok();
  };
  ASSERT_TRUE(eio::run_graph({"a", "b", "c"}, deps, pool, work, nullptr).ok());
  EXPECT_EQ(ran, (std::vector<std::string>{"a", "b", "c"}));
}

TEST(RunGraph, IndependentNodesOverlapInTime) {
  eio::ThreadPool pool(2);
  auto work = [](const std::string&) {
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    return eio::Status::Ok();
  };
  eio::RunReport report;
  ASSERT_TRUE(eio::run_graph({"left", "right"}, {}, pool, work, &report).ok());
  ASSERT_EQ(report.nodes.size(), 2u);
  const auto& x = report.nodes[0];
  const auto& y = report.nodes[1];
  EXPECT_LT(std::max(x.start_ms, y.start_ms), std::min(x.end_ms, y.end_ms))
      << "runs did not overlap";
  EXPECT_LT(report.total_ms, 280.0);  // well under 2 x 150 ms
}

TEST(RunGraph, StopsAfterAFailure) {
  const Dependencies deps{{"b", {"a"}}};
  eio::ThreadPool pool(2);
  std::atomic<bool> b_ran{false};
  auto work = [&](const std::string& id) {
    if (id == "a") return eio::Status(eio::Code::BackendError, "a broke");
    b_ran = true;
    return eio::Status::Ok();
  };
  const eio::Status status = eio::run_graph({"a", "b"}, deps, pool, work, nullptr);
  EXPECT_EQ(status.message(), "a broke");
  EXPECT_FALSE(b_ran.load());
}
