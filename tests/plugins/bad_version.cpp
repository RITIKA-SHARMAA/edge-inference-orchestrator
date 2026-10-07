// A plugin built against a different (future) plugin API version.
#include "eio/eio_plugin.h"

namespace {
const eio_backend_api kApi = {EIO_PLUGIN_API_VERSION + 1,
                              "bad_version",
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr};
}

extern "C" EIO_PLUGIN_EXPORT const eio_backend_api* eio_plugin_get_api(void) { return &kApi; }
