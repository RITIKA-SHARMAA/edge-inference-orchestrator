"""Tests for the ctypes bindings. Need a build (and, for MobileNetV2, the downloaded models)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import eio  # noqa: E402

DATA = ROOT / "tests" / "data"


@pytest.fixture
def runtime():
    with eio.Runtime() as rt:
        yield rt


def test_version_and_backends(runtime):
    assert eio.version() == "0.1.0"
    assert runtime.backends == ["cpu", "mock"]


def test_session_describes_and_runs_a_model(runtime):
    session = runtime.session(DATA / "add_one.onnx")
    assert session.inputs == {"x": (-1, 4)}
    out = session.run({"x": np.arange(8, dtype=np.float32).reshape(2, 4)})
    np.testing.assert_array_equal(out["y"], np.arange(1, 9, dtype=np.float32).reshape(2, 4))


def test_inputs_are_converted_to_float32(runtime):
    out = runtime.session(DATA / "add_one.onnx").run({"x": [[1, 2, 3, 4]]})
    assert out["y"].dtype == np.float32
    np.testing.assert_array_equal(out["y"], [[2, 3, 4, 5]])


def test_errors_become_python_exceptions(runtime):
    session = runtime.session(DATA / "add_one.onnx")
    with pytest.raises(eio.EioError, match=r"invalid argument: .*\[1, 5\]"):
        session.run({"x": np.zeros((1, 5), dtype=np.float32)})
    with pytest.raises(eio.EioError, match="not found"):
        runtime.session("model.onnx", backend="gpu")


def test_pipeline_runs_and_reports_timings(runtime):
    pipeline = runtime.pipeline(DATA / "parallel.json", threads=2)
    out = pipeline.run({"x": np.ones((1, 4), dtype=np.float32)})
    np.testing.assert_array_equal(out["result"], [[2, 2, 2, 2]])
    assert {name for name, _, _ in pipeline.timings()} == {"left", "right", "join"}


def test_missing_plugin_folder():
    with pytest.raises(eio.EioError, match="not found"):
        eio.Runtime("no/such/folder")


@pytest.mark.skipif(not (ROOT / "models" / "mobilenetv2.onnx").exists(), reason="run get_models.py")
def test_mobilenet_recognises_the_cat(runtime):
    logits = runtime.session(ROOT / "models" / "mobilenetv2.onnx").run(
        {"input": np.load(ROOT / "samples" / "cat.npy")})["output"]
    assert 281 <= int(logits.argmax()) <= 285  # ImageNet domestic cat classes
