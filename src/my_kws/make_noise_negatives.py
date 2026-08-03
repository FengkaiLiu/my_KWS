"""Generate pure-noise negative samples for training.

Run from project root (after precompute_features):
    uv run python -m my_kws.make_noise_negatives

Why: the model's training negatives were all *spoken words*. Pure background
noise never appeared as a standalone sample, so continuous-stream evaluation
showed 336 FA/hour on noise-only audio — the model has no concept of
"nobody is speaking". Standard fix (same idea as the official Speech
Commands 'silence' class): random 1 s crops of _background_noise_ as
label-0 samples.

Implementation detail: crops are written straight into the feature cache
as data/cache/logmel/_noise_/noise_NNNN.npy. CachedKeywordSpottingDataset
picks them up with zero code changes — directory name not in positive_words
=> label 0; filenames not in validation/testing lists => all assigned to
train (val/test stay untouched for comparability with previous runs).

Note: train.py's compute_pos_weight walks the raw wav tree and won't count
these crops, so pos_weight stays 25.3 while the true neg count grows
slightly. With 4,000 crops on 81,615 negatives the drift is ~5% — ignored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from .features import log_mel_spectrogram

SR = 16000


def main() -> None:
    parser = argparse.ArgumentParser(description="Make noise negatives")
    parser.add_argument("--data-root", default="data/raw/speech_commands")
    parser.add_argument("--cache-root", default="data/cache/logmel")
    parser.add_argument("--n-crops", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    noise_dir = Path(args.data_root) / "_background_noise_"
    out_dir = Path(args.cache_root) / "_noise_"
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = []
    for f in sorted(noise_dir.glob("*.wav")):
        audio, sr = sf.read(f, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        assert sr == SR, f"{f.name}: expected {SR} Hz, got {sr}"
        if len(audio) > SR:
            sources.append((f.stem, audio))
    print(f"{len(sources)} noise sources")

    for i in range(args.n_crops):
        name, audio = sources[rng.integers(len(sources))]
        start = rng.integers(0, len(audio) - SR)
        crop = audio[start:start + SR] * rng.uniform(0.1, 1.0)  # vary level
        feat = log_mel_spectrogram(crop).astype(np.float32)
        np.save(out_dir / f"noise_{i:04d}.npy", feat)
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{args.n_crops}")

    print(f"done: {args.n_crops} noise negatives in {out_dir}")
    print("next: retrain with --use-cache, then rerun the streaming benchmark")


if __name__ == "__main__":
    main()
