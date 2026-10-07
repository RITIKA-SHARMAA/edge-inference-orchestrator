/* The contract between eio_core and a backend plugin.
 *
 * A backend is a shared library (eio_backend_<name>.so / .dylib / .dll) that
 * exports ONE function, eio_plugin_get_api(), returning a table of function
 * pointers. eio_core finds the library at runtime with dlopen/LoadLibrary, looks
 * that symbol up, checks api_version, and from then on only calls through the
 * table. Plain C types only, so a plugin can be built by any compiler.
 */
#ifndef EIO_PLUGIN_H
#define EIO_PLUGIN_H

#include <stddef.h>
#include <stdint.h>

#include "eio_export.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Bump when the table below changes in an incompatible way. */
#define EIO_PLUGIN_API_VERSION 1u
#define EIO_MAX_RANK 8
#define EIO_NAME_LEN 128

typedef enum eio_dtype {
  EIO_DTYPE_UNSUPPORTED = 0,
  EIO_DTYPE_FLOAT32 = 1,
  EIO_DTYPE_INT64 = 7
} eio_dtype;

/* Describes one model input or output. A dimension of -1 means "any size". */
typedef struct eio_io_desc {
  char name[EIO_NAME_LEN];
  int64_t shape[EIO_MAX_RANK];
  size_t rank;
  eio_dtype dtype;
} eio_io_desc;

/* A float32 tensor passed across the boundary. For inputs the caller owns
 * name/data; for outputs the plugin allocates them and frees them in
 * free_outputs(). */
typedef struct eio_plugin_tensor {
  const char* name;
  int64_t shape[EIO_MAX_RANK];
  size_t rank;
  float* data;
  size_t count; /* number of float elements */
} eio_plugin_tensor;

/* All functions return 0 on success. On failure they write a message into err. */
typedef struct eio_backend_api {
  uint32_t api_version;
  const char* name;
  int (*load)(const char* model_path, void** model, char* err, size_t err_len);
  void (*unload)(void* model);
  size_t (*io_count)(void* model, int is_output);
  int (*io_desc)(void* model, int is_output, size_t index, eio_io_desc* desc);
  int (*run)(void* model, const eio_plugin_tensor* inputs, size_t n_inputs,
             eio_plugin_tensor** outputs, size_t* n_outputs, char* err, size_t err_len);
  void (*free_outputs)(eio_plugin_tensor* outputs, size_t n_outputs);
} eio_backend_api;

typedef const eio_backend_api* (*eio_plugin_get_api_fn)(void);

#define EIO_PLUGIN_ENTRY_SYMBOL "eio_plugin_get_api"

#ifdef __cplusplus
}
#endif

#endif /* EIO_PLUGIN_H */
