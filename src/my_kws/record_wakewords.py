"""Record wake-word samples from the microphone.

Run from project root:
    uv run python -m my_kws.record_wakewords --word yes --n 20
    uv run python -m my_kws.record_wakewords --word hey_computer --n 40

Saves 16 kHz mono wavs to data/raw/my_wakewords/<word>/<word>_NNN.wav,
Speech-Commands-compatible layout so the existing dataset code can point
positive_dir / positive_words at it later.

Collection tips (this is the "data collection" part of the job posting):
    - vary distance to mic, speaking speed, and intonation between takes
    - record in at least two rooms / noise conditions
    - a few deliberately quiet and a few loud takes
    - also record NEGATIVES: similar-sounding words ("yesterday", "less"),
      to later measure confusability
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def main() -> None:
    parser = argparse.ArgumentParser(description="Record wake-word samples")
    parser.add_argument("--word", required=True,
                        help="label, e.g. yes / hey_computer")
    parser.add_argument("--n", type=int, default=20, help="number of takes")
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--out", default="data/raw/my_wakewords")
    parser.add_argument("--device", type=int, default=None,
                        help="sounddevice input device index (see --list)")
    parser.add_argument("--list", action="store_true",
                        help="list audio devices and exit")
    args = parser.parse_args()

    import sounddevice as sd

    if args.list:
        print(sd.query_devices())
        return

    out_dir = Path(args.out) / args.word
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.wav")))
    print(f"saving to {out_dir}  (found {existing} existing takes, "
          f"numbering continues)")

    n_samples = int(args.seconds * SR)
    for i in range(args.n):
        idx = existing + i
        input(f"\n[{i + 1}/{args.n}] press Enter, then say "
              f"\"{args.word}\" after the beep...")
        print("  3..", end="", flush=True); time.sleep(0.4)
        print(" 2..", end="", flush=True); time.sleep(0.4)
        print(" 1..", end="", flush=True); time.sleep(0.4)
        print(" GO", flush=True)

        rec = sd.rec(n_samples, samplerate=SR, channels=1,
                     dtype="float32", device=args.device)
        sd.wait()
        audio = rec.squeeze(1)

        peak = float(np.abs(audio).max())
        rms = float(np.sqrt(np.mean(audio ** 2)))
        path = out_dir / f"{args.word}_{idx:03d}.wav"
        sf.write(path, audio, SR)
        flag = ""
        if peak < 0.02:
            flag = "  <-- very quiet, check mic / re-take?"
        elif peak > 0.99:
            flag = "  <-- clipped, move back / re-take?"
        print(f"  saved {path.name}  peak={peak:.3f} rms={rms:.4f}{flag}")

    print(f"\ndone: {args.n} takes in {out_dir}")


if __name__ == "__main__":
    main()
