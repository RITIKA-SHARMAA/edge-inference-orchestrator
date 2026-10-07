// ThreadPool: a fixed number of worker threads taking tasks from one queue.
//
// Creating a thread per task is slow and unbounded; a pool reuses N threads and
// caps how much runs at once. submit() never blocks; the destructor waits for
// queued tasks to finish and joins the workers.
#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace eio {

class ThreadPool {
 public:
  explicit ThreadPool(size_t threads);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  void submit(std::function<void()> task);
  size_t size() const { return workers_.size(); }

 private:
  void worker_loop();

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable wake_;
  bool stopping_ = false;
};

}  // namespace eio
