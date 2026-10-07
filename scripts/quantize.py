"""Create INT8 versions of both models with ONNX Runtime's quantization tools.

MobileNetV2 -> STATIC quantization. A CNN's activations have ranges that can be
  measured ahead of time, so we run calibration images through the model, record the
  ranges, and bake fixed scales in. Weights and activations are both INT8 at runtime.
  QDQ format (QuantizeLinear/DequantizeLinear pairs) and per-channel weight scales,
  which usually keep accuracy closer to FP32 for convolutions.
  Calibration method: Percentile (99.999th) instead of MinMax. MinMax lets a few extreme
  activation values stretch the INT8 range and waste precision; clipping them helped
  here (see docs/QUANTIZATION-NOTES.md and scripts/calibration_experiment.py).

digits MLP -> DYNAMIC quantization. Weights are converted to INT8 ahead of time;
  activation ranges are computed on the fly for each input. No calibration data needed,
  and it suits small fully-connected (MatMul) models.

Run after prepare_data.py: python scripts/quantize.py
"""

import numpy as np
import onnx
from onnx import version_converter
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from common import DATA, MODELS


class CalibrationImages(CalibrationDataReader):
    """Feeds calibration crops one at a time, as ONNX Runtime expects."""

    def __init__(self, path):
        self.batches = iter(np.load(path)[:, None])  # [N, 1, 3, 224, 224]

    def get_next(self):
        batch = next(self.batches, None)
        return None if batch is None else {"input": batch}


if __name__ == "__main__":
    # Per-channel scales need DequantizeLinear's `axis` attribute, which exists from
    # opset 13. The model zoo file is opset 12, so convert it first.
    opset13 = MODELS / "mobilenetv2.opset13.onnx"
    onnx.save(version_converter.convert_version(onnx.load(MODELS / "mobilenetv2.onnx"), 13), opset13)

    # Shape inference + graph clean-up, as ONNX Runtime recommends before quantizing.
    prepared = MODELS / "mobilenetv2.prep.onnx"
    quant_pre_process(str(opset13), str(prepared))
    quantize_static(
        str(prepared),
        str(MODELS / "mobilenetv2.int8.onnx"),
        CalibrationImages(DATA / "mobilenet" / "calib.npy"),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.Percentile,
    )
    prepared.unlink()
    opset13.unlink()

    quantize_dynamic(
        str(MODELS / "digits_mlp.onnx"),
        str(MODELS / "digits_mlp.int8.onnx"),
        weight_type=QuantType.QInt8,
    )
    for name in ["mobilenetv2", "digits_mlp"]:
        fp32 = (MODELS / f"{name}.onnx").stat().st_size
        int8 = (MODELS / f"{name}.int8.onnx").stat().st_size
        print(f"{name}: {fp32 / 1e6:.2f} MB -> {int8 / 1e6:.2f} MB")
