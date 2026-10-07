/* Public C API of the Edge Inference Orchestrator.
 *
 * Why C and not C++: C++ compilers differ in name mangling, class layout and
 * exception handling, so a C++ class cannot safely cross a DLL boundary or be
 * called from Python. A C API with opaque handles is stable: callers only ever
 * hold pointers to structs they cannot see inside.
 *
 * Rules:
 *  - Every function that can fail returns an eio_code; eio_last_error() gives
 *    the message for the calling thread.
 *  - No C++ exception ever escapes these functions.
 *  - Whoever creates a handle destroys it with the matching *_destroy().
 *    Sessions and pipelines borrow backends from their runtime, so destroy them
 *    before the runtime.
 *  - Tensors are float32, row-major.
 */
#ifndef EIO_H
#define EIO_H

#include <stddef.h>
#include <stdint.h>

#include "eio_export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum eio_code {
  EIO_OK = 0,
  EIO_INVALID_ARGUMENT = 1,
  EIO_NOT_FOUND = 2,
  EIO_PLUGIN_ERROR = 3,
  EIO_BACKEND_ERROR = 4,
  EIO_PIPELINE_ERROR = 5,
  EIO_INTERNAL_ERROR = 6
} eio_code;

typedef struct eio_runtime eio_runtime;   /* loaded backends */
typedef struct eio_session eio_session;   /* one loaded model */
typedef struct eio_pipeline eio_pipeline; /* a validated graph of models */
typedef struct eio_outputs eio_outputs;   /* results of one run */

typedef struct eio_input {
  const char* name;
  const int64_t* shape;
  size_t rank;
  const float* data;
  size_t count; /* number of float elements */
} eio_input;

EIO_API const char* eio_version(void);
EIO_API const char* eio_last_error(void);
/* 0 = debug, 1 = info, 2 = warning, 3 = error, 4 = off. Logs are JSON lines on stderr. */
EIO_API void eio_set_log_level(int level);

/* Loads every eio_backend_* library found in plugin_dir. */
EIO_API eio_code eio_runtime_create(const char* plugin_dir, eio_runtime** out);
EIO_API void eio_runtime_destroy(eio_runtime* runtime);
EIO_API size_t eio_runtime_backend_count(const eio_runtime* runtime);
EIO_API const char* eio_runtime_backend_name(const eio_runtime* runtime, size_t index);

EIO_API eio_code eio_session_create(eio_runtime* runtime, const char* backend,
                                    const char* model_path, eio_session** out);
EIO_API void eio_session_destroy(eio_session* session);
EIO_API size_t eio_session_io_count(const eio_session* session, int is_output);
EIO_API eio_code eio_session_io_info(const eio_session* session, int is_output, size_t index,
                                     const char** name, const int64_t** shape, size_t* rank);
EIO_API eio_code eio_session_run(eio_session* session, const eio_input* inputs, size_t n_inputs,
                                 eio_outputs** out);

EIO_API eio_code eio_pipeline_load(eio_runtime* runtime, const char* json_path, size_t threads,
                                   eio_pipeline** out);
EIO_API void eio_pipeline_destroy(eio_pipeline* pipeline);
EIO_API eio_code eio_pipeline_run(eio_pipeline* pipeline, const eio_input* inputs, size_t n_inputs,
                                  eio_outputs** out);
/* Per-node timings of the most recent successful run, in completion order. */
EIO_API size_t eio_pipeline_timing_count(const eio_pipeline* pipeline);
EIO_API eio_code eio_pipeline_timing(const eio_pipeline* pipeline, size_t index, const char** node,
                                     double* start_ms, double* end_ms);

EIO_API size_t eio_outputs_count(const eio_outputs* outputs);
EIO_API eio_code eio_outputs_get(const eio_outputs* outputs, size_t index, const char** name,
                                 const int64_t** shape, size_t* rank, const float** data,
                                 size_t* count);
EIO_API void eio_outputs_destroy(eio_outputs* outputs);

#ifdef __cplusplus
}
#endif

#endif /* EIO_H */
