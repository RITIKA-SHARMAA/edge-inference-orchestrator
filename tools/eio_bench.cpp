// eio_bench: measure one model's latency, size and accuracy, using only the C API.
//
//   eio_bench --model models/mobilenetv2.int8.onnx --data data/mobilenet --tag int8
//   (optional: --runs 200 --warmup 20 --out results/bench.jsonl)
//
// --data is a folder with:
//   X.npy        all samples, shape [N, ...] (one sample = one batch-1 inference)
//   labels.json  {"labels": [...N ints...], "reference": "ground_truth" | "fp32_top1"}
//
// Latency: `warmup` untimed runs, then `runs` timed batch-1 runs cycling through
// the samples. Accuracy: one pass over every sample, argmax compared to the label.
// One JSON line per invocation is appended to --out; scripts/report.py turns the
// lines into SQLite and docs/BENCHMARK.md.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "cli_common.h"
#include "npy.h"

namespace {

double percentile(std::vector<double> values, double p) {
  // Nearest-rank on the sorted samples: the smallest value with at least p% of runs at or below it.
  if (values.empty()) return 0;
  std::sort(values.begin(), values.end());
  const auto rank = static_cast<size_t>(std::ceil(p / 100.0 * static_cast<double>(values.size())));
  return values[std::clamp<size_t>(rank, 1, values.size()) - 1];
}

std::string cpu_name() {
#if defined(__linux__)
  std::ifstream file("/proc/cpuinfo");
  for (std::string line; std::getline(file, line);) {
    if (line.rfind("model name", 0) == 0) return line.substr(line.find(':') + 2);
  }
#elif defined(_WIN32)
  if (const char* id = std::getenv("PROCESSOR_IDENTIFIER")) return id;
#endif
  return "unknown";
}

const char* os_name() {
#if defined(_WIN32)
  return "windows";
#elif defined(__APPLE__)
  return "macos";
#else
  return "linux";
#endif
}

std::string compiler() {
#if defined(__clang__)
  return "clang " __clang_version__;
#elif defined(__GNUC__)
  return "gcc " __VERSION__;
#elif defined(_MSC_VER)
  return "msvc " + std::to_string(_MSC_VER);
#else
  return "unknown";
#endif
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args(argv + 1, argv + argc);
  const std::string model = cli::arg_value(args, "--model");
  const std::string data_dir = cli::arg_value(args, "--data");
  if (model.empty() || data_dir.empty()) {
    std::cerr << "usage: eio_bench --model FILE --data DIR [--tag NAME] [--backend cpu] "
                 "[--warmup 20] [--runs 200] [--out results/bench.jsonl] [--plugins DIR]\n";
    return 2;
  }
  const std::string tag = cli::arg_value(args, "--tag", "model");
  const std::string backend = cli::arg_value(args, "--backend", "cpu");
  const int warmup = std::stoi(cli::arg_value(args, "--warmup", "20"));
  const int runs = std::stoi(cli::arg_value(args, "--runs", "200"));
  const std::string out_path = cli::arg_value(args, "--out", "results/bench.jsonl");

  // Load the dataset.
  eio::Tensor samples;
  eio::Status status = eio::read_npy((std::filesystem::path(data_dir) / "X.npy").string(), samples);
  if (!status.ok()) {
    std::cerr << "error: " << status.message() << "\n";
    return 1;
  }
  std::ifstream labels_file(std::filesystem::path(data_dir) / "labels.json");
  if (!labels_file) {
    std::cerr << "error: missing labels.json in " << data_dir << "\n";
    return 1;
  }
  const auto labels_doc = nlohmann::json::parse(labels_file);
  const auto labels = labels_doc.at("labels").get<std::vector<int64_t>>();
  const size_t n = static_cast<size_t>(samples.shape.at(0));
  if (labels.size() != n) {
    std::cerr << "error: " << n << " samples but " << labels.size() << " labels\n";
    return 1;
  }
  const size_t sample_size = samples.data.size() / n;
  std::vector<int64_t> sample_shape = samples.shape;
  sample_shape[0] = 1;

  // Load the model through the C API.
  eio_runtime* raw_runtime = nullptr;
  const std::string plugin_dir =
      cli::arg_value(args, "--plugins", cli::default_plugin_dir(argv[0]));
  if (eio_runtime_create(plugin_dir.c_str(), &raw_runtime) != EIO_OK) return cli::fail("plugins");
  cli::Runtime runtime(raw_runtime);
  eio_session* raw_session = nullptr;
  if (eio_session_create(runtime.get(), backend.c_str(), model.c_str(), &raw_session) != EIO_OK) {
    return cli::fail("cannot load " + model);
  }
  cli::Session session(raw_session);
  const char* input_name;
  const int64_t* model_shape;
  size_t model_rank;
  eio_session_io_info(session.get(), 0, 0, &input_name, &model_shape, &model_rank);

  auto run_sample = [&](size_t i, std::vector<float>* scores) -> bool {
    eio_input input{input_name, sample_shape.data(), sample_shape.size(),
                    samples.data.data() + i * sample_size, sample_size};
    eio_outputs* raw = nullptr;
    if (eio_session_run(session.get(), &input, 1, &raw) != EIO_OK) return false;
    cli::Outputs outputs(raw);
    if (scores) {
      const char* name;
      const int64_t* shape;
      size_t rank, count;
      const float* values;
      eio_outputs_get(outputs.get(), 0, &name, &shape, &rank, &values, &count);
      scores->assign(values, values + count);
    }
    return true;
  };

  // Accuracy pass: top-1 (best class equals the label) and top-5 (label among the best five).
  size_t correct = 0;
  size_t correct_top5 = 0;
  std::vector<float> scores;
  for (size_t i = 0; i < n; ++i) {
    if (!run_sample(i, &scores)) return cli::fail("inference failed");
    const auto best =
        static_cast<int64_t>(std::max_element(scores.begin(), scores.end()) - scores.begin());
    correct += (best == labels[i]) ? 1 : 0;
    for (size_t idx : cli::top_k(scores.data(), scores.size(), 5)) {
      if (static_cast<int64_t>(idx) == labels[i]) ++correct_top5;
    }
  }

  // Latency pass.
  for (int i = 0; i < warmup; ++i) {
    if (!run_sample(static_cast<size_t>(i) % n, nullptr)) return cli::fail("inference failed");
  }
  std::vector<double> times_ms;
  for (int i = 0; i < runs; ++i) {
    const auto start = std::chrono::steady_clock::now();
    if (!run_sample(static_cast<size_t>(i) % n, nullptr)) return cli::fail("inference failed");
    times_ms.push_back(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
            .count());
  }
  double mean = 0;
  for (double t : times_ms) mean += t;
  mean /= static_cast<double>(times_ms.size());

  const char* threads_env = std::getenv("EIO_ORT_THREADS");
  nlohmann::json result = {
      {"tag", tag},
      {"model", std::filesystem::path(model).filename().string()},
      {"backend", backend},
      {"size_bytes", std::filesystem::file_size(model)},
      {"samples", n},
      {"accuracy", static_cast<double>(correct) / static_cast<double>(n)},
      {"accuracy_top5", static_cast<double>(correct_top5) / static_cast<double>(n)},
      {"reference", labels_doc.value("reference", "ground_truth")},
      {"warmup", warmup},
      {"runs", runs},
      {"p50_ms", percentile(times_ms, 50)},
      {"p95_ms", percentile(times_ms, 95)},
      {"mean_ms", mean},
      {"ort_intra_op_threads", threads_env ? threads_env : "1"},
      {"os", os_name()},
      {"cpu", cpu_name()},
      {"hardware_threads", std::thread::hardware_concurrency()},
      {"compiler", compiler()},
      {"timestamp", static_cast<int64_t>(std::time(nullptr))},
  };

  const auto out_dir = std::filesystem::path(out_path).parent_path();
  if (!out_dir.empty()) std::filesystem::create_directories(out_dir);
  std::ofstream out(out_path, std::ios::app);
  out << result.dump() << "\n";
  std::cout << result.dump(2) << "\n";
  return 0;
}
