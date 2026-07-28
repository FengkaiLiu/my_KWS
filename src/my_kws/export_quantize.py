"""Export, quantize, and validate the KWS model — the edge-deployment toolchain.

Run from the project root:
    uv run python -m my_kws.export_quantize

Pipeline (each stage gates the next — this IS "automated toolchain validation"):
    1. Load best PyTorch checkpoint
    2. Export to ONNX (fp32)
    3. PARITY CHECK: PyTorch vs ONNX outputs must agree to ~1e-5,
       otherwise abort — no point quantizing a broken export
    4. Post-training static quantization to int8, calibrated on REAL
       training features (calibration data must match deployment
       distribution, or activation ranges will be wrong)
    5. Evaluate fp32 ONNX and int8 ONNX on the TEST set at the chosen
       deployment threshold
    6. Benchmark single-sample CPU latency (the edge-relevant number)
    7. Print the size / accuracy / latency comparison table

Outputs:
    models/dscnn_fp32.onnx
    models/dscnn_int8.onnx
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .cached_datasets import CachedKeywordSpottingDataset
from .model import DSCNN


# --------------------------------------------------------------------------
# Stage 2: export
# --------------------------------------------------------------------------

def export_onnx(model: torch.nn.Module, out_path: Path) -> None:
    model.eval()
    dummy = torch.randn(1, 1, 40, 98)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["log_mel"],
        output_names=["logit"],
        dynamic_axes={"log_mel": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
    )
    print(f"[export] wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


# --------------------------------------------------------------------------
# Stage 3: parity check
# --------------------------------------------------------------------------

def parity_check(model: torch.nn.Module, onnx_path: Path,
                 dataset, n_samples: int = 64, tol: float = 1e-4) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    model.eval()

    idx = np.random.default_rng(0).choice(len(dataset), n_samples, replace=False)
    batch = torch.stack([dataset[int(i)][0] for i in idx])          # (N,1,40,98)

    with torch.no_grad():
        ref = model(batch).numpy()
    out = sess.run(["logit"], {"log_mel": batch.numpy()})[0]

    max_diff = float(np.max(np.abs(ref - out)))
    print(f"[parity] PyTorch vs ONNX max |diff| = {max_diff:.2e} "
          f"over {n_samples} real samples")
    assert max_diff < tol, (
        f"parity check FAILED (diff {max_diff} >= {tol}) — "
        "do not proceed to quantization with a broken export"
    )
    print("[parity] PASSED")


# --------------------------------------------------------------------------
# Stage 4: int8 static quantization
# --------------------------------------------------------------------------

def quantize(fp32_path: Path, int8_path: Path, calib_dataset,
             n_calib: int = 256) -> None:
    from onnxruntime.quantization import (
        CalibrationDataReader, QuantFormat, QuantType, quantize_static,
    )

    class RealDataCalibReader(CalibrationDataReader):
        """Feeds real log-mel features so activation ranges match deployment."""

        def __init__(self) -> None:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(calib_dataset), n_calib, replace=False)
            self._it = iter(
                {"log_mel": calib_dataset[int(i)][0].unsqueeze(0).numpy()}
                for i in idx
            )

        def get_next(self):
            return next(self._it, None)

    # optional graph cleanup pass recommended before quantization
    pre_path = fp32_path.with_suffix(".pre.onnx")
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(str(fp32_path), str(pre_path))
        src = pre_path
    except Exception as e:                                  # noqa: BLE001
        print(f"[quant] pre-process skipped ({e}); quantizing raw export")
        src = fp32_path

    quantize_static(
        str(src), str(int8_path),
        calibration_data_reader=RealDataCalibReader(),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    pre_path.unlink(missing_ok=True)
    print(f"[quant] wrote {int8_path} ({int8_path.stat().st_size / 1024:.1f} KB), "
          f"calibrated on {n_calib} real samples")


# --------------------------------------------------------------------------
# Stage 5: test-set evaluation of an ONNX model
# --------------------------------------------------------------------------

def evaluate_onnx(onnx_path: Path, dataset, threshold: float,
                  batch_size: int = 256) -> dict[str, float]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    scores, labels = [], []
    for start in range(0, len(dataset), batch_size):
        xs, ys = zip(*(dataset[i] for i in
                       range(start, min(start + batch_size, len(dataset)))))
        batch = torch.stack(xs).numpy()
        logits = sess.run(["logit"], {"log_mel": batch})[0].squeeze(1)
        scores.append(1.0 / (1.0 + np.exp(-logits)))
        labels.append(np.array(ys, dtype=bool))
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)

    pred = scores >= threshold
    tp = int((pred & labels).sum()); fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return {"precision": p, "recall": r,
            "f1": 2 * p * r / max(p + r, 1e-9), "fp": fp, "fn": fn}


# --------------------------------------------------------------------------
# Stage 6: latency benchmark (single sample, CPU — the edge scenario)
# --------------------------------------------------------------------------

def benchmark(onnx_path: Path, n_runs: int = 300) -> dict[str, float]:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1          # single-core: closest to a DSP budget
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 1, 40, 98).astype(np.float32)

    for _ in range(20):                    # warmup
        sess.run(["logit"], {"log_mel": x})
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(["logit"], {"log_mel": x})
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {"mean_ms": float(times.mean()),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95))}


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export + quantize + validate")
    parser.add_argument("--checkpoint", default="models/dscnn_best_cached.pt")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="deployment threshold chosen on val in notebook 02")
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fp32_path = out_dir / "dscnn_fp32.onnx"
    int8_path = out_dir / "dscnn_int8.onnx"

    # stage 1: load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DSCNN()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[load] {args.checkpoint} (epoch {ckpt['epoch']}, "
          f"val F1 {ckpt['val_f1']:.3f})")

    test_ds = CachedKeywordSpottingDataset(split="test", training=False)
    train_ds = CachedKeywordSpottingDataset(split="train", training=False)

    # stages 2-4
    export_onnx(model, fp32_path)
    parity_check(model, fp32_path, test_ds)
    quantize(fp32_path, int8_path, train_ds)

    # stages 5-7
    print("\n[eval] test set @ threshold", args.threshold)
    results = {}
    for name, path in [("fp32", fp32_path), ("int8", int8_path)]:
        m = evaluate_onnx(path, test_ds, args.threshold)
        lat = benchmark(path)
        results[name] = {**m, **lat,
                         "size_kb": path.stat().st_size / 1024}

    ckpt_kb = Path(args.checkpoint).stat().st_size / 1024
    print(f"\n{'':10s} {'size KB':>8s} {'P':>6s} {'R':>6s} {'F1':>6s} "
          f"{'FP':>4s} {'FN':>4s} {'p50 ms':>7s} {'p95 ms':>7s}")
    print(f"{'torch ckpt':10s} {ckpt_kb:8.1f} {'—':>6s} {'—':>6s} {'—':>6s}")
    for name, r in results.items():
        print(f"{name:10s} {r['size_kb']:8.1f} {r['precision']:6.3f} "
              f"{r['recall']:6.3f} {r['f1']:6.3f} {r['fp']:4d} {r['fn']:4d} "
              f"{r['p50_ms']:7.3f} {r['p95_ms']:7.3f}")

    d_f1 = results["int8"]["f1"] - results["fp32"]["f1"]
    print(f"\nquantization F1 delta: {d_f1:+.4f} | "
          f"size: {results['fp32']['size_kb'] / results['int8']['size_kb']:.1f}x smaller")


if __name__ == "__main__":
    main()
