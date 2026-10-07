// eio_run: run one model or a pipeline from the command line, using only the C API.
//
//   eio_run --list-backends
//   eio_run --model models/mobilenetv2.onnx --input samples/cat.npy --labels
//   models/imagenet_labels.txt eio_run --pipeline pipelines/example.json --input
//   image=samples/cat.npy --threads 2
//           (add --labels models/imagenet_labels.txt for class names)
//
// Common options: --backend NAME (default cpu), --plugins DIR, --top K, --log-level 0..4
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "cli_common.h"

namespace {

void usage() {
  std::cerr
      << "usage:\n"
         "  eio_run --list-backends\n"
         "  eio_run --model FILE [--backend cpu] --input [NAME=]FILE.npy [--labels FILE] [--top "
         "5]\n"
         "  eio_run --pipeline FILE --input NAME=FILE.npy [--threads 2] [--labels FILE] [--top 5]\n"
         "options: --plugins DIR  --log-level 0..4\n";
}

// Prints each output: top-k classes when a label file matches its width, else a preview.
void print_outputs(const eio_outputs* outputs, const std::vector<std::string>& labels, size_t k) {
  for (size_t i = 0; i < eio_outputs_count(outputs); ++i) {
    const char* name;
    const int64_t* shape;
    size_t rank, count;
    const float* data;
    eio_outputs_get(outputs, i, &name, &shape, &rank, &data, &count);
    std::cout << "output '" << name << "' shape [";
    for (size_t d = 0; d < rank; ++d) std::cout << (d ? ", " : "") << shape[d];
    std::cout << "]\n";
    const size_t width = rank ? static_cast<size_t>(shape[rank - 1]) : count;
    if (!labels.empty() && width == labels.size() && count >= width) {
      for (size_t idx : cli::top_k(data, width, k)) {
        std::cout << "  " << std::fixed << std::setprecision(4) << std::setw(10) << data[idx]
                  << "  " << labels[idx] << "\n";
      }
    } else {
      std::cout << " ";
      for (size_t j = 0; j < count && j < 10; ++j) std::cout << " " << data[j];
      std::cout << (count > 10 ? " ...\n" : "\n");
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args(argv + 1, argv + argc);
  if (args.empty() || cli::has_flag(args, "--help")) {
    usage();
    return args.empty() ? 2 : 0;
  }
  eio_set_log_level(std::stoi(cli::arg_value(args, "--log-level", "2")));

  eio_runtime* raw_runtime = nullptr;
  const std::string plugin_dir =
      cli::arg_value(args, "--plugins", cli::default_plugin_dir(argv[0]));
  if (eio_runtime_create(plugin_dir.c_str(), &raw_runtime) != EIO_OK) {
    return cli::fail("cannot load plugins from " + plugin_dir);
  }
  cli::Runtime runtime(raw_runtime);

  if (cli::has_flag(args, "--list-backends")) {
    for (size_t i = 0; i < eio_runtime_backend_count(runtime.get()); ++i) {
      std::cout << eio_runtime_backend_name(runtime.get(), i) << "\n";
    }
    return 0;
  }

  const std::vector<std::string> labels = cli::read_lines(cli::arg_value(args, "--labels"));
  const size_t k = static_cast<size_t>(std::stoul(cli::arg_value(args, "--top", "5")));

  // Inputs are "NAME=file.npy", or just "file.npy" for a single-input model.
  std::vector<cli::NamedInput> inputs;
  for (const std::string& spec : cli::arg_values(args, "--input")) {
    const auto eq = spec.find('=');
    const std::string name = eq == std::string::npos ? "" : spec.substr(0, eq);
    const std::string path = eq == std::string::npos ? spec : spec.substr(eq + 1);
    cli::NamedInput input;
    std::string error;
    if (!cli::load_input(name, path, input, error)) {
      std::cerr << "error: " << error << "\n";
      return 1;
    }
    inputs.push_back(std::move(input));
  }
  if (inputs.empty()) {
    usage();
    return 2;
  }

  const std::string model = cli::arg_value(args, "--model");
  const std::string pipeline_path = cli::arg_value(args, "--pipeline");

  if (!model.empty()) {
    const std::string backend = cli::arg_value(args, "--backend", "cpu");
    eio_session* raw_session = nullptr;
    if (eio_session_create(runtime.get(), backend.c_str(), model.c_str(), &raw_session) != EIO_OK) {
      return cli::fail("cannot load " + model);
    }
    cli::Session session(raw_session);
    if (inputs.size() == 1 && inputs[0].name.empty()) {
      const char* name;
      const int64_t* shape;
      size_t rank;
      eio_session_io_info(session.get(), 0, 0, &name, &shape, &rank);
      inputs[0].name = name;  // default to the model's first input
    }
    std::vector<eio_input> views;
    for (const auto& input : inputs) views.push_back(input.view());
    eio_outputs* raw_outputs = nullptr;
    if (eio_session_run(session.get(), views.data(), views.size(), &raw_outputs) != EIO_OK) {
      return cli::fail("run failed");
    }
    cli::Outputs outputs(raw_outputs);
    print_outputs(outputs.get(), labels, k);
    return 0;
  }

  if (!pipeline_path.empty()) {
    const size_t threads = static_cast<size_t>(std::stoul(cli::arg_value(args, "--threads", "2")));
    eio_pipeline* raw_pipeline = nullptr;
    if (eio_pipeline_load(runtime.get(), pipeline_path.c_str(), threads, &raw_pipeline) != EIO_OK) {
      return cli::fail("cannot load pipeline " + pipeline_path);
    }
    cli::Pipeline pipeline(raw_pipeline);
    std::vector<eio_input> views;
    for (const auto& input : inputs) views.push_back(input.view());
    eio_outputs* raw_outputs = nullptr;
    if (eio_pipeline_run(pipeline.get(), views.data(), views.size(), &raw_outputs) != EIO_OK) {
      return cli::fail("pipeline run failed");
    }
    cli::Outputs outputs(raw_outputs);
    print_outputs(outputs.get(), labels, k);

    std::cout << "\nnode timings (" << threads << " threads, ms since run start)\n";
    std::cout << std::left << std::setw(12) << "node" << std::right << std::setw(10) << "start"
              << std::setw(10) << "end" << std::setw(10) << "took" << "\n";
    for (size_t i = 0; i < eio_pipeline_timing_count(pipeline.get()); ++i) {
      const char* node;
      double start, end;
      eio_pipeline_timing(pipeline.get(), i, &node, &start, &end);
      std::cout << std::left << std::setw(12) << node << std::right << std::fixed
                << std::setprecision(2) << std::setw(10) << start << std::setw(10) << end
                << std::setw(10) << end - start << "\n";
    }
    return 0;
  }

  usage();
  return 2;
}
