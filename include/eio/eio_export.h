/* Symbol export macros.
 *
 * Windows only exports functions from a DLL that are explicitly marked
 * __declspec(dllexport), and a program using the DLL should see them as
 * __declspec(dllimport). On Linux/macOS everything is hidden by default in this
 * project (-fvisibility=hidden), so exported functions are marked "default".
 * EIO_BUILDING_LIBRARY is defined only while compiling eio_core itself.
 */
#ifndef EIO_EXPORT_H
#define EIO_EXPORT_H

#if defined(_WIN32)
#if defined(EIO_BUILDING_LIBRARY)
#define EIO_API __declspec(dllexport)
#else
#define EIO_API __declspec(dllimport)
#endif
#define EIO_PLUGIN_EXPORT __declspec(dllexport)
#else
#define EIO_API __attribute__((visibility("default")))
#define EIO_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

#endif /* EIO_EXPORT_H */
