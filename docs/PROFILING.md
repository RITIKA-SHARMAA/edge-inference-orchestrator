# Profiling: where does the time go?

## Tool used

`perf` was not available in the environment where this was measured, so the profile below
comes from **Valgrind's callgrind**, which counts the CPU instructions executed in every
function (it runs the program about 50x slower, but the *proportions* are what matter).
How to do the same with `perf` on Linux is at the end.

What was profiled: one run of the example pipeline (FP32 + INT8 MobileNetV2 and a
`softmax_mean` step) with one worker thread, including loading both models:

```bash
valgrind --tool=callgrind --callgrind-out-file=cg.out \
  build/bin/eio_run --pipeline pipelines/example.json --input image=samples/cat.npy --threads 1
callgrind_annotate --inclusive=yes cg.out | less
```

## Result (529.5 million instructions in total)

| Part of the program | Share of instructions | Where |
|---|---|---|
| Loading the two models (`eio_pipeline_load` → `ModelSession::create` → ONNX Runtime session creation and graph optimisation) | **62.4%** | ONNX Runtime |
| Running the two models (`ModelSession::run` → CPU plugin `run` → ONNX Runtime) | **32.4%** | ONNX Runtime |
| Copying each node's input tensors inside `Pipeline::run_node` | 0.25% | this project |
| Converting C API inputs to tensors (`to_tensors`) | 0.11% | this project |
| Built-in `softmax_mean` step | 0.02% | this project |
| Scheduling: `run_graph`, Kahn's sort, thread pool submit | < 0.01% | this project |
| Loading the plugin libraries (`dlopen`) and reading the `.npy` input | about 0.6% | this project / OS |

## What it means

1. **Model loading dominates a single run.** Creating an ONNX Runtime session parses the model
   and optimises its graph, which costs almost twice as much as running it. That is why the
   SDK loads each model once (when a session or pipeline is created) and reuses it for every
   run. `eio_bench` measures latency only after loading and warm-up for the same reason.
2. **The orchestration layer is cheap.** Everything this project adds around ONNX Runtime
   (validation, the scheduler, the thread pool, the C API, the combine step) is well under
   1% of the work. Speeding up inference means changing the model (e.g. INT8) or the backend,
   not the scheduler.
3. **The biggest overhead of our own is copying.** Each pipeline node gets its own copy of its
   input tensors (the 600 KB image is copied once per branch). Sharing read-only tensors
   (`std::shared_ptr<const Tensor>`) would remove those copies. Not done yet, because at
   0.25% it is not worth the extra complexity.

## With perf (Linux)

```bash
perf record -g build/bin/eio_bench --model models/mobilenetv2.onnx --data data/mobilenet \
  --runs 500 --warmup 20 --out /tmp/perf-bench.jsonl
perf report --sort dso,symbol
```

Expect most samples inside `libonnxruntime.so` convolution kernels during the timed runs.
