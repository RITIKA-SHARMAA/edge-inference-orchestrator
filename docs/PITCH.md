# 60-second pitch

> "I built a small cross-platform inference orchestrator in C++17, to learn how an on-device AI
> SDK works.
>
> It loads ONNX models through ONNX Runtime and exposes everything through a C API, built as a
> shared library: `.so` on Linux, `.dylib` on macOS, `.dll` on Windows, so any language can call it.
> I use the same API from Python with ctypes.
>
> Backends are plugins: separate libraries the core loads at runtime with `dlopen` or
> `LoadLibrary`, each exporting one versioned table of functions. The tests build deliberately
> broken plugins to check they are rejected cleanly.
>
> Pipelines are JSON graphs. On load I validate every connection and shape and run Kahn's
> algorithm to order the nodes and catch cycles; at run time a scheduler starts each node on a
> thread pool as soon as its inputs are ready, so independent branches run in parallel.
>
> I quantized MobileNetV2 to INT8 with static per-channel calibration. It came out 72.5% smaller
> and about 1.9 times faster at the median, 4.8 against 9.0 milliseconds on a cloud CPU, agreeing
> with the FP32 model on 85.7% of top-1 predictions and keeping its answer in the top 5 every time.
> A tiny MLP actually got slower in INT8, which taught me quantization only pays off when compute
> dominates.
>
> It builds and runs its 50 C++ tests on GCC, Clang and MSVC in CI, plus a sanitizer build."

## If they ask "what would you do next?"

- Add an NPU or GPU backend as another plugin, which is exactly what the plugin table is for.
- Share read-only tensors between pipeline nodes instead of copying them (profiling shows copies are
  the largest overhead of my own code, though still under 1%).
- Evaluate on a labelled ImageNet subset instead of agreement with FP32.
