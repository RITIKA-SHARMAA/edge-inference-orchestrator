#include "thread_pool.h"

namespace eio {

ThreadPool::ThreadPool(size_t threads) {
  if (threads == 0) threads = 1;
  for (size_t i = 0; i < threads; ++i) workers_.emplace_back([this] { worker_loop(); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  wake_.notify_all();
  for (auto& worker : workers_) worker.join();
}

void ThreadPool::submit(std::function<void()> task) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.push(std::move(task));
  }
  wake_.notify_one();
}

void ThreadPool::worker_loop() {
  for (;;) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      // Sleep until there is work or we are shutting down (spurious wake-ups are re-checked).
      wake_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
      if (stopping_ && tasks_.empty()) return;
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    task();  // run outside the lock so other workers can take tasks meanwhile
  }
}

}  // namespace eio
