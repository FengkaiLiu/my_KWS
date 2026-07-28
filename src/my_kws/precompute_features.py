"""Precompute log-mel features for the whole dataset -> data/cache/logmel/.

Run once from the project root:
    uv run python -m my_kws.precompute_features

Why cache?
    During training, ~90% of wall time is CPU recomputing the same
    STFT + mel filterbank for the same files every epoch. Precomputing
    turns each __getitem__ into a single np.load.

The tradeoff (this IS the notebook-02 experiment):
    Waveform-level augmentations (time shift, add noise) need the raw
    waveform, so cached training loses them. SpecAugment operates on the
    log-mel itself, so it survives and stays random per epoch.
    We measure: how much speed do we gain, and how much F1 (if any)
    do we lose by dropping waveform augmentation?

Storage: 40*98 float32 = ~15.7 KB/file, ~105k files => ~1.6 GB total.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .audio_io import load_audio
from .features import log_mel_spectrogram


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute log-mel features")
    parser.add_argument("--root", default="data/raw/speech_commands")
    parser.add_argument("--out", default="data/cache/logmel")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--length-s", type=float, default=1.0)
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)

    wavs = [
        f for d in sorted(root.iterdir())
        if d.is_dir() and d.name != "_background_noise_"
        for f in sorted(d.glob("*.wav"))
    ]
    print(f"found {len(wavs):,} wav files")

    t0 = time.time()
    done = skipped = 0
    for i, wav_path in enumerate(wavs, 1):
        out_path = out_root / wav_path.parent.name / (wav_path.stem + ".npy")
        if out_path.exists():          # resumable: rerun skips finished files
            skipped += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)

        waveform = load_audio(wav_path, args.sr, True, args.length_s)
        feat = log_mel_spectrogram(waveform).astype(np.float32)   # (40, 98)
        np.save(out_path, feat)
        done += 1

        if i % 5000 == 0:
            rate = i / (time.time() - t0)
            eta = (len(wavs) - i) / rate
            print(f"{i:,}/{len(wavs):,}  ({rate:.0f} files/s, ETA {eta/60:.1f} min)")

    elapsed = time.time() - t0
    print(f"done: {done:,} computed, {skipped:,} skipped, {elapsed/60:.1f} min")
    print(f"cache at: {out_root.resolve()}")


if __name__ == "__main__":
    main()
