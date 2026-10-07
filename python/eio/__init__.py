"""Python bindings for the Edge Inference Orchestrator, over its C API with ctypes.

ctypes loads the shared library (libeio_core.so / .dylib / eio_core.dll) and calls the
plain C functions declared in include/eio/eio.h. Each function's argument and return
types are declared below, so ctypes converts Python values correctly.

    import numpy as np, eio
    with eio.Runtime() as rt:
        with rt.session("models/mobilenetv2.onnx") as s:
            out = s.run({"input": np.load("samples/cat.npy")})
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

__all__ = ["EioError", "Runtime", "Session", "Pipeline", "set_log_level", "version"]

_ROOT = Path(__file__).resolve().parents[2]
_CODES = {1: "invalid argument", 2: "not found", 3: "plugin error", 4: "backend error",
          5: "pipeline error", 6: "internal error"}


class EioError(RuntimeError):
    """A non-zero eio_code, with the message from eio_last_error()."""

    def __init__(self, code: int, message: str):
        super().__init__(f"{_CODES.get(code, code)}: {message}")
        self.code = code


def _build_dir() -> Path:
    return Path(os.environ.get("EIO_BUILD_DIR", _ROOT / "build")) / "bin"


def _library_path() -> Path:
    if "EIO_LIB" in os.environ:
        return Path(os.environ["EIO_LIB"])
    name = {"win32": "eio_core.dll", "darwin": "libeio_core.dylib"}.get(sys.platform, "libeio_core.so")
    return _build_dir() / name


class _Input(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("rank", ctypes.c_size_t), ("data", ctypes.POINTER(ctypes.c_float)),
                ("count", ctypes.c_size_t)]


def _load() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_library_path()))
    P = ctypes.c_void_p
    sigs = {
        "eio_version": ([], ctypes.c_char_p),
        "eio_last_error": ([], ctypes.c_char_p),
        "eio_set_log_level": ([ctypes.c_int], None),
        "eio_runtime_create": ([ctypes.c_char_p, ctypes.POINTER(P)], ctypes.c_int),
        "eio_runtime_destroy": ([P], None),
        "eio_runtime_backend_count": ([P], ctypes.c_size_t),
        "eio_runtime_backend_name": ([P, ctypes.c_size_t], ctypes.c_char_p),
        "eio_session_create": ([P, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(P)], ctypes.c_int),
        "eio_session_destroy": ([P], None),
        "eio_session_io_count": ([P, ctypes.c_int], ctypes.c_size_t),
        "eio_session_io_info": ([P, ctypes.c_int, ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p),
                                 ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),
                                 ctypes.POINTER(ctypes.c_size_t)], ctypes.c_int),
        "eio_session_run": ([P, ctypes.POINTER(_Input), ctypes.c_size_t, ctypes.POINTER(P)], ctypes.c_int),
        "eio_pipeline_load": ([P, ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(P)], ctypes.c_int),
        "eio_pipeline_destroy": ([P], None),
        "eio_pipeline_run": ([P, ctypes.POINTER(_Input), ctypes.c_size_t, ctypes.POINTER(P)], ctypes.c_int),
        "eio_pipeline_timing_count": ([P], ctypes.c_size_t),
        "eio_pipeline_timing": ([P, ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p),
                                 ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)],
                                ctypes.c_int),
        "eio_outputs_count": ([P], ctypes.c_size_t),
        "eio_outputs_get": ([P, ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p),
                             ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)), ctypes.POINTER(ctypes.c_size_t),
                             ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(ctypes.c_size_t)],
                            ctypes.c_int),
        "eio_outputs_destroy": ([P], None),
    }
    for name, (args, result) in sigs.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    return lib


_lib = _load()


def _check(code: int) -> None:
    if code != 0:
        raise EioError(code, (_lib.eio_last_error() or b"").decode())


def version() -> str:
    return _lib.eio_version().decode()


def set_log_level(level: int) -> None:
    """0 debug, 1 info, 2 warning, 3 error, 4 off. Logs are JSON lines on stderr."""
    _lib.eio_set_log_level(level)


def _inputs(arrays: dict[str, np.ndarray]) -> tuple[ctypes.Array, list]:
    """Build the C input array. `keep` holds the numpy buffers alive during the call."""
    keep, items = [], []
    for name, array in arrays.items():
        data = np.ascontiguousarray(array, dtype=np.float32)
        shape = np.asarray(data.shape, dtype=np.int64)
        keep += [data, shape, name.encode()]
        items.append(_Input(keep[-1], shape.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), data.ndim,
                            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), data.size))
    return (_Input * len(items))(*items), keep


def _outputs(handle: ctypes.c_void_p) -> dict[str, np.ndarray]:
    """Copy every output into numpy arrays, then free the C-side results."""
    try:
        result = {}
        for i in range(_lib.eio_outputs_count(handle)):
            name, shape, data = ctypes.c_char_p(), ctypes.POINTER(ctypes.c_int64)(), ctypes.POINTER(ctypes.c_float)()
            rank, count = ctypes.c_size_t(), ctypes.c_size_t()
            _check(_lib.eio_outputs_get(handle, i, ctypes.byref(name), ctypes.byref(shape), ctypes.byref(rank),
                                        ctypes.byref(data), ctypes.byref(count)))
            dims = tuple(shape[d] for d in range(rank.value))
            result[name.value.decode()] = np.ctypeslib.as_array(data, shape=(count.value,)).copy().reshape(dims)
        return result
    finally:
        _lib.eio_outputs_destroy(handle)


class _Handle:
    _destroy = ""

    def __init__(self, handle: ctypes.c_void_p):
        self._handle = handle

    def close(self) -> None:
        # getattr: __del__ can run on an object whose __init__ failed before _handle was set.
        if getattr(self, "_handle", None):
            getattr(_lib, self._destroy)(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()


class Session(_Handle):
    """One loaded model. Create through Runtime.session()."""

    _destroy = "eio_session_destroy"

    def __init__(self, handle, runtime: "Runtime"):
        super().__init__(handle)
        self._runtime = runtime  # sessions borrow the runtime's backends: keep it alive

    def _io(self, is_output: int) -> dict[str, tuple[int, ...]]:
        io = {}
        for i in range(_lib.eio_session_io_count(self._handle, is_output)):
            name, shape, rank = ctypes.c_char_p(), ctypes.POINTER(ctypes.c_int64)(), ctypes.c_size_t()
            _check(_lib.eio_session_io_info(self._handle, is_output, i, ctypes.byref(name),
                                            ctypes.byref(shape), ctypes.byref(rank)))
            io[name.value.decode()] = tuple(shape[d] for d in range(rank.value))
        return io

    @property
    def inputs(self) -> dict[str, tuple[int, ...]]:
        return self._io(0)

    @property
    def outputs(self) -> dict[str, tuple[int, ...]]:
        return self._io(1)

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        array, _keep = _inputs(inputs)
        out = ctypes.c_void_p()
        _check(_lib.eio_session_run(self._handle, array, len(inputs), ctypes.byref(out)))
        return _outputs(out)


class Pipeline(_Handle):
    """A validated graph of models. Create through Runtime.pipeline()."""

    _destroy = "eio_pipeline_destroy"

    def __init__(self, handle, runtime: "Runtime"):
        super().__init__(handle)
        self._runtime = runtime

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        array, _keep = _inputs(inputs)
        out = ctypes.c_void_p()
        _check(_lib.eio_pipeline_run(self._handle, array, len(inputs), ctypes.byref(out)))
        return _outputs(out)

    def timings(self) -> list[tuple[str, float, float]]:
        """(node, start_ms, end_ms) for the last run, in completion order."""
        result = []
        for i in range(_lib.eio_pipeline_timing_count(self._handle)):
            node, start, end = ctypes.c_char_p(), ctypes.c_double(), ctypes.c_double()
            _check(_lib.eio_pipeline_timing(self._handle, i, ctypes.byref(node), ctypes.byref(start), ctypes.byref(end)))
            result.append((node.value.decode(), start.value, end.value))
        return result


class Runtime(_Handle):
    """Loads every backend plugin in `plugin_dir` (default: <build>/bin/plugins)."""

    _destroy = "eio_runtime_destroy"

    def __init__(self, plugin_dir: str | os.PathLike | None = None):
        plugin_dir = Path(plugin_dir or os.environ.get("EIO_PLUGIN_DIR", _build_dir() / "plugins"))
        handle = ctypes.c_void_p()
        _check(_lib.eio_runtime_create(str(plugin_dir).encode(), ctypes.byref(handle)))
        super().__init__(handle)
        self._children: list[_Handle] = []

    @property
    def backends(self) -> list[str]:
        return [_lib.eio_runtime_backend_name(self._handle, i).decode()
                for i in range(_lib.eio_runtime_backend_count(self._handle))]

    def session(self, model_path: str | os.PathLike, backend: str = "cpu") -> Session:
        handle = ctypes.c_void_p()
        _check(_lib.eio_session_create(self._handle, backend.encode(), str(model_path).encode(), ctypes.byref(handle)))
        session = Session(handle, self)
        self._children.append(session)
        return session

    def pipeline(self, json_path: str | os.PathLike, threads: int = 2) -> Pipeline:
        handle = ctypes.c_void_p()
        _check(_lib.eio_pipeline_load(self._handle, str(json_path).encode(), threads, ctypes.byref(handle)))
        pipeline = Pipeline(handle, self)
        self._children.append(pipeline)
        return pipeline

    def close(self) -> None:
        # Sessions and pipelines must be destroyed before the runtime that owns their backends.
        for child in getattr(self, "_children", []):
            child.close()
        super().close()
