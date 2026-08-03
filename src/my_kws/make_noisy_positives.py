"""Generate noise-mixed POSITIVE samples for training.

Run from project root:
    uv run python -m my_kws.make_noisy_positives

Why: adding pure-noise negatives fixed FA/hour (336 -> 0) but crashed
streaming recall (0.850 -> 0.417). Root cause: the cached pipeline dropped
waveform-level add_noise augmentation, so every training positive is CLEAN
"yes" — while streamed keywords sit on background noise. The model learned
"noise features => 0" and now suppresses noisy positives.

Fix: mix training-split "yes" files with _background_noise_ at random SNRs
and write the features into data/cache/logmel/yes/ as extra .npy files.
Zero dataset-code changes (same trick as _noise_): directory "yes" => label
1, synthetic filenames absent from validation/testing lists => train split.

LEAKAGE GUARD: only wavs NOT in validation_list.txt / testing_list.txt are
augmented. Augmenting a val/test file into the training set would leak.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio_io import load_audio
from .augment import add_noise
from .features import log_mel_spectrogram

SR = 16000


def main() -> None:
    parser = argparse.ArgumentParser(description="Make noisy positives")
    parser.add_argument("--data-root", default="data/raw/speech_commands")
    parser.add_argument("--cache-root", default="data/cache/logmel")
    parser.add_argument("--word", default="yes")
    parser.add_argument("--variants", type=int, default=2,
                        help="noisy copies per training-split positive")
    parser.add_argument("--snr-low", type=float, default=0.0)
    parser.add_argument("--snr-high", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    root = Path(args.data_root)
    out_dir = Path(args.cache_root) / args.word
    assert out_dir.exists(), "run precompute_features first"
    rng = np.random.default_rng(args.seed)

    # leakage guard: collect held-out relative paths
    held_out: set[str] = set()
    for name in ("validation_list.txt", "testing_list.txt"):
        held_out.update(
            line.strip().replace("\\", "/")
            for line in (root / name).read_text().splitlines()
            if line.strip()
        )

    train_wavs = [
        f for f in sorted((root / args.word).glob("*.wav"))
        if f"{args.word}/{f.name}" not in held_out
    ]
    print(f"{len(train_wavs)} training-split '{args.word}' files "
          f"(held-out excluded)")

    # load noise sources once
    noises = []
    for f in sorted((root / "_background_noise_").glob("*.wav")):
        audio, _ = sf.read(f, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        noises.append(audio)

    count = 0
    for wav_path in train_wavs:
        clean = load_audio(wav_path, SR, True, 1.0)
        for v in range(args.variants):
            noise = noises[rng.integers(len(noises))]
            start = rng.integers(0, len(noise) - SR)
            snr = rng.uniform(args.snr_low, args.snr_high)
            noisy = add_noise(clean, noise[start:start + SR], snr_db=snr)
            feat = log_mel_spectrogram(noisy).astype(np.float32)
            np.save(out_dir / f"aug_{wav_path.stem}_v{v}.npy", feat)
            count += 1
        if count % 1000 < args.variants:
            print(f"{count} written...")

    print(f"done: {count} noisy positives in {out_dir}")
    print("note: pos_weight in train.py counts raw wavs only; true positive "
          "count is now ~3x that, so effective pos_weight ~8 — this is fine "
          "(the imbalance genuinely shrank), but note it in the run log.")
    print("next: retrain --use-cache, re-sweep threshold (nb02), re-export, "
          "rerun streaming")


if __name__ == "__main__":
    main()
