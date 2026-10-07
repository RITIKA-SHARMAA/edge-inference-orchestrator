#include "scheduler.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <set>

namespace eio {

namespace {
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// dependents["a"] = nodes that need "a" (the reverse of deps).
std::map<std::string, std::vector<std::string>> invert(const std::vector<std::string>& nodes,
                                                       const Dependencies& deps) {
  std::map<std::string, std::vector<std::string>> dependents;
  for (const auto& node : nodes) dependents[node];
  for (const auto& [node, needs] : deps) {
    for (const auto& dep : needs) dependents[dep].push_back(node);
  }
  return dependents;
}
}  // namespace

Status order_nodes(const std::vector<std::string>& nodes, const Dependencies& deps,
                   std::vector<std::string>& order) {
  std::set<std::string> known(nodes.begin(), nodes.end());
  std::map<std::string, size_t> in_degree;
  for (const auto& node : nodes) {
    auto it = deps.find(node);
    const size_t n = (it == deps.end()) ? 0 : it->second.size();
    in_degree[node] = n;
    if (it != deps.end()) {
      for (const auto& dep : it->second) {
        if (!known.count(dep)) {
          return {Code::PipelineError, "node '" + node + "' depends on unknown node '" + dep + "'"};
        }
      }
    }
  }
  const auto dependents = invert(nodes, deps);

  // Start from nodes with no dependencies, in declaration order for a stable result.
  std::queue<std::string> ready;
  for (const auto& node : nodes) {
    if (in_degree[node] == 0) ready.push(node);
  }
  order.clear();
  while (!ready.empty()) {
    std::string node = ready.front();
    ready.pop();
    order.push_back(node);
    for (const auto& next : dependents.at(node)) {
      if (--in_degree[next] == 0) ready.push(next);
    }
  }

  if (order.size() != nodes.size()) {
    std::string stuck;
    for (const auto& node : nodes) {
      if (in_degree[node] > 0) stuck += (stuck.empty() ? "" : ", ") + node;
    }
    return {Code::PipelineError, "pipeline has a cycle involving: " + stuck};
  }
  return Status::Ok();
}

Status run_graph(const std::vector<std::string>& nodes, const Dependencies& deps, ThreadPool& pool,
                 const std::function<Status(const std::string&)>& work, RunReport* report) {
  std::vector<std::string> order;
  Status status = order_nodes(nodes, deps, order);
  if (!status.ok()) return status;

  const auto dependents = invert(nodes, deps);
  std::map<std::string, size_t> remaining;  // unfinished dependencies per node
  for (const auto& node : nodes) {
    auto it = deps.find(node);
    remaining[node] = (it == deps.end()) ? 0 : it->second.size();
  }

  // Shared state, always accessed under `mutex`.
  std::mutex mutex;
  std::condition_variable done_cv;
  size_t in_flight = 0;
  size_t finished = 0;
  Status first_error;
  std::vector<NodeTiming> timings;
  const auto run_start = Clock::now();

  // submit() is always called with `mutex` held.
  std::function<void(const std::string&)> submit = [&](const std::string& node) {
    ++in_flight;
    pool.submit([&, node] {
      NodeTiming timing{node, ms_since(run_start), 0};
      Status result = work(node);  // the actual model run happens here, outside the lock
      timing.end_ms = ms_since(run_start);

      std::lock_guard<std::mutex> lock(mutex);
      --in_flight;
      ++finished;
      timings.push_back(timing);
      if (!result.ok()) {
        if (first_error.ok()) first_error = result;
      } else if (first_error.ok()) {
        for (const auto& next : dependents.at(node)) {
          if (--remaining[next] == 0) submit(next);
        }
      }
      done_cv.notify_all();
    });
  };

  {
    std::unique_lock<std::mutex> lock(mutex);
    for (const auto& node : order) {
      if (remaining[node] == 0) submit(node);
    }
    // Finished when every node ran, or when an error occurred and nothing is still running.
    done_cv.wait(lock,
                 [&] { return finished == nodes.size() || (!first_error.ok() && in_flight == 0); });
  }

  if (report) {
    report->nodes = timings;
    report->total_ms = ms_since(run_start);
  }
  return first_error;
}

}  // namespace eio
