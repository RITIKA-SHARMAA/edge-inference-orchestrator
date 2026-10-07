"""Create the tiny ONNX models used by the C++ tests (committed under tests/data).

add_one.onnx:   y = x + 1, x is float32 [batch, 4]   (dynamic batch dimension)
int64_ids.onnx: y = ids, ids is int64 [1, 4]         (the SDK must reject it)

Run: python scripts/make_test_models.py
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).resolve().parent.parent / "tests" / "data"


def add_one() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 4])
    one = numpy_helper.from_array(np.ones((1, 4), dtype=np.float32), name="one")
    node = helper.make_node("Add", ["x", "one"], ["y"])
    graph = helper.make_graph([node], "add_one", [x], [y], initializer=[one])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


def int64_ids() -> onnx.ModelProto:
    ids = helper.make_tensor_value_info("ids", TensorProto.INT64, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.INT64, [1, 4])
    node = helper.make_node("Identity", ["ids"], ["y"])
    graph = helper.make_graph([node], "int64_ids", [ids], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, model in {"add_one.onnx": add_one(), "int64_ids.onnx": int64_ids()}.items():
        onnx.checker.check_model(model)
        onnx.save(model, OUT / name)
        print("wrote", OUT / name)
