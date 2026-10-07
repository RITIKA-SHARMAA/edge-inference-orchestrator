"""Benchmark FP32 vs INT8 for both models with the C++ eio_bench tool, then write the report.

Run after quantize.py and a build:
  python scripts/benchmark.py [--build-dir build] [--runs 300] [--warmup 30]

Each eio_bench call appends one JSON line to results/bench.jsonl. report.py then loads
all lines into SQLite (results/bench.db) and writes docs/BENCHMARK.md from the latest run.
"""

import argparse
import subprocess
import sys
import uuid

from common import DATA, MODELS, RESULTS, ROOT

RUNS = [
    ("mobilenetv2", "fp32", MODELS / "mobilenetv2.onnx", DATA / "mobilenet"),
    ("mobilenetv2", "int8", MODELS / "mobilenetv2.int8.onnx", DATA / "mobilenet"),
    ("digits_mlp", "fp32", MODELS / "digits_mlp.onnx", DATA / "digits"),
    ("digits_mlp", "int8", MODELS / "digits_mlp.int8.onnx", DATA / "digits"),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default=str(ROOT / "build"))
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()

    exe = ROOT / args.build_dir / "bin" / ("eio_bench.exe" if sys.platform == "win32" else "eio_bench")
    RESULTS.mkdir(exist_ok=True)
    session = uuid.uuid4().hex[:8]  # groups the lines of this benchmark session
    for model, precision, path, data in RUNS:
        print(f"== {model} {precision}")
        subprocess.run(
            [str(exe), "--model", str(path), "--data", str(data), "--tag", f"{model}:{precision}:{session}",
             "--runs", str(args.runs), "--warmup", str(args.warmup),
             "--out", str(RESULTS / "bench.jsonl")],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    subprocess.run([sys.executable, str(ROOT / "scripts" / "report.py"), "--session", session], check=True)
