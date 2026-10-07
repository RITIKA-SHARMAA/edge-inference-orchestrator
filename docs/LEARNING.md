# Learning guide

This explains the project part by part, in the order it was built, using this code as the
example. Each part ends with likely interview questions and short answers.

---

## Part 1: building it (CMake, compilers, dependencies)

**Files:** `CMakeLists.txt`, `cmake/dependencies.cmake`, `cmake/warnings.cmake`

- **CMake vs Make.** `CMakeLists.txt` describes *what* to build (libraries, executables, how they
  link). CMake then *generates* the real build files for each platform: Makefiles or Ninja files on
  Linux/macOS, Visual Studio projects on Windows. Make (or Ninja, or MSBuild) runs those files.
  That is how one project builds with GCC, Clang and MSVC.
- **Targets in this project:**
  - `eio_internal`: all the C++ code, as a static library (copied into whatever links it).
  - `eio_core`: the shared library users link. It contains only `c_api.cpp` plus `eio_internal`.
  - `eio_backend_cpu`, `eio_backend_mock`: plugins, built as `MODULE` libraries (made to be loaded
    at runtime, never linked).
  - `eio_run`, `eio_bench`: command-line tools that use only the C API.
  - `eio_unit_tests`, `eio_c_api_tests`: GoogleTest executables.
- **Dependencies** are pinned and downloaded with `FetchContent`: the prebuilt ONNX Runtime 1.20.1
  archive for the current OS, nlohmann/json 3.11.3 and GoogleTest 1.15.2. ONNX Runtime becomes an
  `IMPORTED` target, so the CPU plugin links it like any other library.
- **Output layout:** everything runnable goes to `build/bin`, plugins to `build/bin/plugins`. The
  `$<0:>` generator expression stops Visual Studio adding a `Release/` folder, so paths are the same
  on every OS.
- **Warnings:** `-Wall -Wextra -Wpedantic -Wshadow -Wconversion` on GCC/Clang, `/W4` on MSVC. The
  project builds with zero warnings on GCC 13 and Clang 18.

**Interview questions**

1. *Why build `eio_internal` as a static library and `eio_core` as a shared one?* The shared
   library hides every C++ symbol and exports only the C API. Tests need to reach the C++
   classes directly, so they link the static library that holds the same code.
2. *What is the difference between `SHARED` and `MODULE`?* Both produce a `.so`/`.dll`. `SHARED`
   is meant to be linked at build time; `MODULE` is only ever opened at runtime with
   `dlopen`/`LoadLibrary`, which is what a plugin is.
3. *How does the plugin find `libonnxruntime` at runtime?* On Linux/macOS through an RPATH
   set on the plugin. On Windows a post-build step copies `onnxruntime.dll` next to the plugin,
   and the loader uses `LOAD_WITH_ALTERED_SEARCH_PATH` so Windows looks in the plugin's folder.

---

## Part 2: running one model (ONNX, ONNX Runtime, ModelSession)

**Files:** `src/plugins/cpu_backend.cpp`, `src/core/model_session.*`, `src/core/tensor.*`, `src/core/npy.*`

- **ONNX** is a file format for trained models: a graph of operators (Conv, Relu, MatMul...) plus
  weights. **ONNX Runtime** is the engine that executes it. Qualcomm's QNN and SNPE play the same
  role for Snapdragon hardware.
- **Inference** means running a trained model on new input. Nothing is trained here at runtime.
- **Tensor** = an array of numbers with a **shape**. MobileNetV2 takes `[batch, 3, 224, 224]`:
  a batch of images, 3 colour channels, 224x224 pixels. `-1` (or a name like `batch_size`) means
  that dimension can be any size.
- **What happens when a model is loaded** (`ModelSession::create`):
  1. The backend creates an `Ort::Session`: ONNX Runtime parses the file and optimises the graph
     (`ORT_ENABLE_ALL` fuses operations such as Conv + BatchNorm + Relu).
  2. The session asks the backend for every input and output: name, shape, element type.
  3. Anything that is not float32 is rejected immediately with a clear message.
- **What happens on `run`:**
  1. `validate()` checks that every model input is present, no unknown input is passed, each shape
     fits (`-1` matches anything) and the data length matches the shape. A wrong shape fails here
     with "input 'x' has shape [1, 5], model expects [-1, 4]", before ONNX Runtime is called.
  2. Input buffers are wrapped as `Ort::Value` tensors **without copying**.
  3. ONNX Runtime runs; outputs are copied into `malloc`ed memory owned by the plugin.
  4. The core copies them into its own tensors and calls the plugin's `free_outputs`, because
     memory must be freed by the library that allocated it (each DLL can have its own heap on Windows).
- **Inputs as `.npy`:** Python prepares images (resize, crop, normalise, channels-first) and saves
  them as NumPy files; `npy.cpp` reads the small header (dtype, order, shape) and the raw floats.
  That keeps image decoding out of the C++ code.

**Interview questions**

1. *Why validate shapes yourself when ONNX Runtime would fail anyway?* To fail early with a message
   that names the input and both shapes, instead of an error from deep inside the runtime.
2. *What does `-1` in a shape mean?* A dynamic dimension, usually batch size.
3. *Why does the plugin copy outputs into `malloc` memory?* The core and plugin may use different
   allocators or C runtimes; returning plain memory plus a `free_outputs` function keeps allocation
   and freeing in the same library.
4. *Why set ONNX Runtime to 1 intra-op thread?* So parallelism comes from the pipeline scheduler and
   benchmark numbers are stable. `EIO_ORT_THREADS` changes it.

---

## Part 3: shared library, C API and plugins

**Files:** `include/eio/eio.h`, `include/eio/eio_export.h`, `include/eio/eio_plugin.h`,
`src/core/c_api.cpp`, `src/core/dynamic_library.*`, `src/core/backend.*`

- **Static vs shared library.** A static library is copied into each program at build time. A
  shared library (`.so` on Linux, `.dylib` on macOS, `.dll` on Windows) is a separate file loaded
  when the program starts or later, shared by every program that uses it, and replaceable without
  rebuilding them.
- **Why a C API in front of C++.** C++ compilers mangle names differently (`eio::Pipeline::run`
  becomes a compiler-specific symbol), lay out classes differently, and handle exceptions
  differently. A C function named `eio_pipeline_run` taking pointers and integers looks the same to
  every compiler and to Python's `ctypes`. `extern "C"` turns name mangling off.
- **Opaque handles.** Callers get `eio_session*` but never see what is inside. The struct can change
  without breaking callers, which keeps the **ABI** (the binary interface) stable.
- **No exceptions across the boundary.** Every exported function runs its body inside `guard()`,
  which catches all exceptions and returns `EIO_INTERNAL_ERROR`. An exception escaping into C or
  Python is undefined behaviour.
- **Exporting symbols.** Windows only exports functions marked `__declspec(dllexport)`, and code
  using the DLL should see `__declspec(dllimport)`. The project builds with `-fvisibility=hidden` on
  Linux/macOS, so it behaves the same way: only functions marked `EIO_API` are visible. Check with
  `nm -D --defined-only build/bin/libeio_core.so`: it lists the 20 `eio_*` functions and, on Linux,
  one standard-library template instantiation that the compiler emits as a weak symbol. No class
  or function of this project's C++ code is exported.
- **Plugins.** Each backend exports one function, `eio_plugin_get_api()`, returning a table of
  function pointers (`load`, `run`, `free_outputs`...) and an `api_version`. `DynamicLibrary` opens
  the file with `dlopen` (Linux/macOS) or `LoadLibraryExW` (Windows) and finds the function with
  `dlsym` / `GetProcAddress`. `Backend::load` then rejects a missing symbol, a null table, a wrong
  version or a missing function pointer. The tests build two deliberately broken plugins to prove it.
- `RTLD_NOW` makes `dlopen` resolve every symbol immediately, so a plugin with a missing dependency
  fails at load time, not in the middle of a run.

**Interview questions**

1. *What does `dlopen` do?* Maps the shared library into the process, loads its dependencies, runs
   its static initialisers and returns a handle; `dlsym` then looks up a symbol's address by name.
2. *Why version the plugin API?* A plugin built against an older or newer table layout would call
   or be called through the wrong function pointers. Checking `api_version` turns that crash into a
   clear error.
3. *What would break the ABI here?* Changing the layout of a struct the caller sees (`eio_input`,
   `eio_backend_api`), or the signature of an exported function. Opaque handles avoid exposing most
   layouts.
4. *How do you know nothing else is exported?* `-fvisibility=hidden` plus explicit `EIO_API`, and
   `--exclude-libs,ALL` for code linked in from the static library; `nm -D` shows the 20 `eio_*`
   C functions and no project C++ symbols.

---

## Part 4: pipelines, the graph and the scheduler

**Files:** `src/core/pipeline.*`, `src/core/scheduler.*`, `src/core/thread_pool.*`,
`src/core/builtin_ops.*`, `pipelines/example.json`

- A **pipeline** is a **DAG** (directed acyclic graph): nodes are model runs or built-in steps; an
  edge means one node's output feeds another's input. In the JSON, a source is either a pipeline
  input (`"image"`) or `"<node>.<output>"` (`"fp32.output"`).
- **Validation at load time** (`Pipeline::validate`):
  1. Every source refers to a real pipeline input or node; dependencies are collected per node.
  2. **Kahn's algorithm** (`order_nodes`) produces an execution order. Start with nodes that depend
     on nothing; each time one is taken, reduce the count of its dependents; a dependent whose count
     reaches zero becomes ready. If some nodes never reach zero, they are part of a **cycle**, and
     the error names them. Cost: O(V + E).
  3. Walking in that order, every model input must be connected, every connection's declared shape
     must be compatible with what the model expects, and built-in steps get matching input shapes.
- **Thread pool:** a fixed number of worker threads sleeping on a condition variable and taking
  tasks from one queue. Creating a thread per task would be slower and unbounded.
- **Scheduler** (`run_graph`): every node with no unfinished dependencies is submitted to the pool.
  When a node finishes, under a mutex, its dependents' counters go down, and any that reach zero
  are submitted. The main thread waits on a condition variable until every node has finished, or
  until an error has happened and nothing is still running. The model itself runs **outside** the
  lock, so independent branches really overlap.
- In the example, `fp32` and `int8` both depend only on the input, so with 2 threads they run at the
  same time; `combine` (softmax each, then average) waits for both.

**Interview questions**

1. *How do you detect a cycle?* Kahn's algorithm: if nodes remain whose dependency count never hit
   zero, they form (or depend on) a cycle.
2. *Why a condition variable and not a loop that checks a flag?* A loop burns CPU; a condition
   variable sleeps until notified, and the predicate is re-checked to handle spurious wake-ups.
3. *What is protected by the mutex in `run_graph`?* The dependency counters, the finished/in-flight
   counts, the first error and the timing list. Not the model run.
4. *How do the tests prove parallelism?* Two mock nodes that each sleep 150 ms must overlap in time,
   and the whole run must take well under 300 ms.
5. *What happens when a node fails?* No new nodes are scheduled, running ones finish, and the first
   error is returned.

---

## Part 5: quantization and benchmarking

**Files:** `scripts/quantize.py`, `scripts/calibration_experiment.py`, `tools/eio_bench.cpp`,
`scripts/benchmark.py`, `scripts/report.py`, `docs/BENCHMARK.md`, `docs/QUANTIZATION-NOTES.md`

- **Quantization** stores numbers as 8-bit integers instead of 32-bit floats:
  `real_value ≈ scale × (int8_value − zero_point)`. Weights become about 4x smaller, and hardware
  with fast INT8 instructions (like an NPU) runs much faster, at some cost in accuracy.
- **Static** quantization (used for MobileNetV2) measures activation ranges ahead of time by running
  **calibration** images through the model, then fixes the scales. **Dynamic** quantization (used for
  the digits MLP) converts weights ahead of time and computes activation scales per input, with no
  calibration data.
- **QDQ format:** the INT8 model contains `QuantizeLinear`/`DequantizeLinear` pairs around operators;
  ONNX Runtime fuses them into INT8 kernels. **Per-channel** weights get one scale per output channel
  instead of one per tensor, which keeps convolutions more accurate. Per-channel needs opset 13, so
  the model is converted from opset 12 first.
- **Calibration method:** MinMax uses the full observed range of each activation; Percentile clips
  the most extreme 0.001%. In this project Percentile raised top-1 agreement from 76.8% to 85.7% on
  the unrelated calibration images (see `docs/QUANTIZATION-NOTES.md`; the evaluation set is small).
- **Benchmark method** (`eio_bench`): one accuracy pass over every sample; then 30 warm-up runs that
  are not timed (first runs pay for memory allocation and cold caches); then 300 timed batch-1 runs.
  It reports **p50** (median) and **p95** (95% of runs finish within this), because an average hides
  slow outliers. Results go to a JSON line, then `report.py` loads them into SQLite and writes the
  markdown table.

**Results** (one machine, see `docs/BENCHMARK.md`): MobileNetV2 INT8 is 72.5% smaller, p50 4.84 ms vs
9.04 ms, top-1 agreement with FP32 85.7%, and FP32's class is in INT8's top 5 on every crop. The digits
MLP gets *slower* in INT8 (0.014 vs 0.008 ms) because quantize/dequantize overhead outweighs the
saving on such a small model, with identical 97.1% accuracy.

**Interview questions**

1. *What does quantization trade?* Precision for size and speed. Speed only improves if the hardware
   has fast INT8 paths and the model is large enough for compute to dominate.
2. *Why was the digits model slower in INT8?* It is tiny (64 inputs, two small layers), so the extra
   quantize/dequantize work costs more than the faster 8-bit maths saves.
3. *Why report p95?* Users feel the slow tail; the median hides it.
4. *How did you measure INT8 accuracy without labels?* As agreement with the FP32 model's top-1 and
   top-5 predictions, and the report states that clearly.
5. *What did the calibration experiment teach you?* My first guess (that calibration images similar
   to the test photos matter most) was not supported; the method mattered more, and with 56 samples
   a few points is only one or two images, so it is not conclusive.

---

## Part 6: Python bindings, logging, profiling, sanitizers

**Files:** `python/eio/__init__.py`, `src/core/log.*`, `docs/PROFILING.md`, `tests/lsan.supp`

- **ctypes** loads `libeio_core.so` / `.dylib` / `eio_core.dll`. Each C function gets its argument
  and return types declared (`argtypes`, `restype`) so Python converts values correctly. NumPy arrays
  are passed by pointer to their data (made contiguous float32 first). Results are copied into new
  NumPy arrays, then the C results are freed. Errors become `eio.EioError` with the C message.
- **Structured logging:** one JSON object per line on stderr (`ts_ms`, `level`, `event`, fields),
  written under a mutex so lines from different threads never mix. Level from `EIO_LOG_LEVEL` or
  `eio_set_log_level()`.
- **Profiling:** callgrind shows model loading is 62% and inference 32% of a one-shot run, and the
  orchestration layer under 1% (details in `docs/PROFILING.md`).
- **Sanitizers:** the CI job builds with AddressSanitizer (out-of-bounds, use-after-free, leaks) and
  UndefinedBehaviorSanitizer. ONNX Runtime itself keeps a few small global allocations alive; that
  was confirmed by loading only `libonnxruntime` without project code, and is suppressed in
  `tests/lsan.supp`. Leaks in this project's code still fail the build.

**Interview questions**

1. *How does Python call your C++?* Through the C API with ctypes; it never sees C++ symbols.
2. *Why does the Python `Runtime` close its sessions first?* Sessions borrow backends owned by the
   runtime; destroying the runtime first would leave sessions pointing at unloaded code.
3. *Where does the time go?* Mostly ONNX Runtime: loading (graph optimisation) and convolution kernels.
4. *What does AddressSanitizer catch?* Out-of-bounds reads and writes, use-after-free, double free and
   leaks, by instrumenting every memory access at compile time.
