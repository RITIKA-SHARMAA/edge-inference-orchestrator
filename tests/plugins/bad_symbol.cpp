// A plugin that forgets to export eio_plugin_get_api. Loading it must fail cleanly.
#include "eio/eio_export.h"

extern "C" EIO_PLUGIN_EXPORT int some_other_function(void) { return 42; }
