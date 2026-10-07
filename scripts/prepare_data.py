"""Prepare MobileNetV2 inputs as .npy files from the sample images bundled with scikit-image.

  samples/cat.npy            one centre-cropped photo of a cat, [1, 3, 224, 224]
  data/mobilenet/calib.npy   48 crops from 6 images, used only to calibrate INT8 quantization
  data/mobilenet/X.npy       56 crops from 7 *different* images, used to evaluate
  data/mobilenet/labels.json the FP32 model's top-1 class for each evaluation crop

There is no labelled ImageNet set here, so INT8 accuracy is measured as agreement with
the FP32 model (how often both pick the same top-1 class). labels.json records this as
reference "fp32_top1" and the benchmark report says so.

Run after get_models.py: python scripts/prepare_data.py
"""

import json

import numpy as np
import onnxruntime as ort
import skimage.data as skdata

from common import DATA, MODELS, SAMPLES, center_crop, random_crop, to_rgb, to_tensor

EVAL_IMAGES = ["chelsea", "coffee", "rocket", "astronaut", "cat", "colorwheel", "immunohistochemistry"]
CALIB_IMAGES = ["hubble_deep_field", "retina", "logo", "coins", "camera", "page"]
CROPS_PER_IMAGE = 8


def crops(names: list[str], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tensors = []
    for name in names:
        image = to_rgb(getattr(skdata, name)())
        tensors += [to_tensor(random_crop(image, rng)) for _ in range(CROPS_PER_IMAGE)]
    return np.stack(tensors).astype(np.float32)


if __name__ == "__main__":
    SAMPLES.mkdir(exist_ok=True)
    cat = to_tensor(center_crop(to_rgb(skdata.chelsea())))[None]
    np.save(SAMPLES / "cat.npy", cat)

    out = DATA / "mobilenet"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "calib.npy", crops(CALIB_IMAGES, seed=1))
    X = crops(EVAL_IMAGES, seed=2)
    np.save(out / "X.npy", X)

    session = ort.InferenceSession(str(MODELS / "mobilenetv2.onnx"), providers=["CPUExecutionProvider"])
    labels = [int(session.run(None, {"input": x[None]})[0].argmax()) for x in X]
    (out / "labels.json").write_text(json.dumps({"labels": labels, "reference": "fp32_top1"}))
    print(f"wrote {len(X)} evaluation crops, {len(labels)} reference labels")
