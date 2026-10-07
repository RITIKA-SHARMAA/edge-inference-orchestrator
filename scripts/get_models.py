"""Download MobileNetV2 and its labels, and train the small digits classifier.

Creates (all git-ignored):
  models/mobilenetv2.onnx        ImageNet classifier from the ONNX model zoo (opset 12)
  models/imagenet_labels.txt     1000 class names, one per line
  models/digits_mlp.onnx         scikit-learn MLP on 8x8 handwritten digits, exported to ONNX
  data/digits/X.npy, labels.json held-out test split with true labels

Run: python scripts/get_models.py
"""

import hashlib
import json
import urllib.request

import numpy as np
import onnx
from onnx.utils import extract_model
from skl2onnx import to_onnx
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DATA, MODELS

ZOO = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification"
MOBILENET_URL = f"{ZOO}/mobilenet/model/mobilenetv2-12.onnx"
MOBILENET_SHA256 = "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5"
LABELS_URL = "https://raw.githubusercontent.com/onnx/models/main/validated/vision/classification/synset.txt"


def download(url: str, dest, sha256: str | None = None) -> None:
    if dest.exists() and (sha256 is None or hashlib.sha256(dest.read_bytes()).hexdigest() == sha256):
        print("have", dest.name)
        return
    print("downloading", url)
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    if sha256 and hashlib.sha256(data).hexdigest() != sha256:
        raise SystemExit(f"checksum mismatch for {url}")
    dest.write_bytes(data)


def imagenet_labels() -> None:
    dest = MODELS / "imagenet_labels.txt"
    if dest.exists():
        print("have", dest.name)
        return
    with urllib.request.urlopen(LABELS_URL, timeout=60) as response:
        lines = response.read().decode().splitlines()
    # "n02123045 tabby, tabby cat" -> "tabby, tabby cat"
    dest.write_text("\n".join(line.split(" ", 1)[1] for line in lines) + "\n")


def digits_model() -> None:
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    model = make_pipeline(
        StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600, random_state=0)
    )
    model.fit(X_train, y_train)
    print(f"digits MLP test accuracy (scikit-learn): {model.score(X_test, y_test):.4f}")

    # Export with a float32 input named "input" and plain probability arrays (no zipmap).
    onx = to_onnx(
        model,
        X_train[:1],
        options={id(model.steps[-1][1]): {"zipmap": False}},
        target_opset=17,
        final_types=None,
    )
    full = MODELS / "digits_mlp.full.onnx"
    onnx.save(onx, full)
    # The SDK moves float32 tensors only, so keep just the float "probabilities" output
    # and drop the int64 "label" output.
    extract_model(str(full), str(MODELS / "digits_mlp.onnx"),
                  [onx.graph.input[0].name], ["probabilities"])
    full.unlink()

    out = DATA / "digits"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", X_test)
    (out / "labels.json").write_text(
        json.dumps({"labels": [int(v) for v in y_test], "reference": "ground_truth"})
    )


if __name__ == "__main__":
    MODELS.mkdir(exist_ok=True)
    download(MOBILENET_URL, MODELS / "mobilenetv2.onnx", MOBILENET_SHA256)
    imagenet_labels()
    digits_model()
    print("done")
