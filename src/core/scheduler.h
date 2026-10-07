// Scheduler: runs a dependency graph on a thread pool.
//
// It knows nothing about models. Each node is just an id, a list of the nodes it
// depends on, and a function to call. A node is submitted to the pool as soon as
// every node it depends on has finished, so independent branches run in parallel.
//
// order_nodes() is Kahn's algorithm: repeatedly take nodes with no unfinished
// dependencies. Anything left over at the end is part of a cycle.
#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "status.h"
#include "thread_pool.h"

namespace eio {

struct NodeTiming {
  std::string id;
  double start_ms = 0;  // relative to the start of the run
  double end_ms = 0;
};

struct RunReport {
  std::vector<NodeTiming> nodes;  // in completion order
  double total_ms = 0;
};

using Dependencies = std::map<std::string, std::vector<std::string>>;  // node -> nodes it needs

// Returns a valid execution order, or a PipelineError naming the nodes in a cycle.
Status order_nodes(const std::vector<std::string>& nodes, const Dependencies& deps,
                   std::vector<std::string>& order);

// Runs every node once. Stops scheduling new nodes after the first failure, waits for
// running ones to finish, and returns that failure.
Status run_graph(const std::vector<std::string>& nodes, const Dependencies& deps, ThreadPool& pool,
                 const std::function<Status(const std::string&)>& work, RunReport* report);

}  // namespace eio
