import sys
from pathlib import Path
import numpy as np

# Compatibility patch for newer ONNX releases
import onnx
import onnx.helper

if not hasattr(onnx, "mapping"):
    onnx.mapping = getattr(onnx, "_mapping", None)

if not hasattr(onnx.helper, "split_complex_to_pairs"):
    onnx.helper.split_complex_to_pairs = lambda *args, **kwargs: None

from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as rt
import lightgbm as lgb

MODEL_DIR = Path("models")


def export():
    lgb_path = MODEL_DIR / "credit_model.txt"
    onnx_path = MODEL_DIR / "credit_model.onnx"

    if not lgb_path.exists():
        sys.exit(f"ERROR: {lgb_path} not found. Run training/train.py first.")

    print(f"Loading LightGBM model from {lgb_path}...")
    booster = lgb.Booster(model_file=str(lgb_path))
    n_features = booster.num_feature()
    print(f"  Features: {n_features}")

    print("Converting to ONNX...")
    initial_types = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_lightgbm(booster, initial_types=initial_types, target_opset=12)

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  Saved: {onnx_path}")

    # Smoke test
    print("\nRunning smoke test...")
    session = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy = np.random.rand(1, n_features).astype(np.float32)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: dummy})
    print(f"  Output shapes: {[o.shape if hasattr(o, 'shape') else type(o) for o in outputs]}")
    prob = outputs[1][0][1] if isinstance(outputs[1], list) else outputs[1][1]
    print(f"  Dummy prediction probability: {float(prob):.4f}")
    print("\nONNX export complete.")


if __name__ == "__main__":
    export()
