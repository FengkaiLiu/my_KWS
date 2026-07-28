"""
datasets.py — Chunk 1 integration.

Ties together io.py (audio loading), features.py (log-mel), and augment.py
(noise / time-shift / SpecAugment) into a PyTorch Dataset for keyword spotting.

Task: binary wake-word detection ("hey computer" vs. not).
Since the real wake-word recordings don't exist yet, the positive class is
sourced from a configurable list of Speech Commands words (default ["yes"]).
When record_wakewords.py is ready, point `positive_dir` at the recorded clips
and set positive_words=None — no other code needs to change.

Layout expected under data/raw/ (produced by download_data.py):

    data/raw/speech_commands/
        yes/ no/ up/ down/ ...            # 35 word folders of .wav
        _background_noise_/               # long noise clips (noise source)
        validation_list.txt               # official val split
        testing_list.txt                  # official test split
    data/raw/esc50/ESC-50-master/
        audio/                            # 2000 environmental .wav
        meta/esc50.csv
"""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio_io import load_audio        # relative import within the my_kws package
from . import features
from . import augment


# Fixed audio geometry (matches features.py / notebook conventions)
SAMPLE_RATE = 16000
CLIP_SECONDS = 1.0            # Speech Commands clips are ~1 s
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400


def _read_split_list(txt_path: Path) -> set[str]:
    """Read a Speech Commands split file into a set of 'word/file.wav' keys."""
    if not txt_path.exists():
        return set()
    with open(txt_path, "r") as f:
        return {line.strip() for line in f if line.strip()}


class KeywordSpottingDataset(Dataset):
    """
    Binary keyword-spotting dataset built on Google Speech Commands v2.

    Each item is (feature, label):
        feature: torch.FloatTensor, shape (1, N_MELS, n_frames) = (1, 40, 98)
                 a log-mel spectrogram with a leading channel dim for CNNs.
        label:   torch.LongTensor scalar, 1 = wake word, 0 = not.

    Splits use the official validation_list.txt / testing_list.txt so no
    speaker leaks across train/val/test.
    """

    def __init__(
        self,
        root: str | Path = "data/raw/speech_commands",
        split: str = "train",                 # "train" | "val" | "test"
        positive_words: list[str] | None = ("yes",),
        positive_dir: str | Path | None = None,
        training: bool = None,                # None -> infer from split
        # augmentation toggles (only used when training=True)
        use_time_shift: bool = True,
        use_add_noise: bool = True,
        use_spec_augment: bool = True,
        max_shift_samples: int = 1600,        # 0.1 s at 16 kHz
        snr_db_range: tuple[float, float] = (5.0, 20.0),
        seed: int | None = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.positive_words = set(positive_words) if positive_words else set()
        self.positive_dir = Path(positive_dir) if positive_dir else None

        # By convention, only the training split gets augmentation.
        self.training = (split == "train") if training is None else training

        self.use_time_shift = use_time_shift
        self.use_add_noise = use_add_noise
        self.use_spec_augment = use_spec_augment
        self.max_shift_samples = max_shift_samples
        self.snr_db_range = snr_db_range

        self._rng = random.Random(seed)

        self.val_list = _read_split_list(self.root / "validation_list.txt")
        self.test_list = _read_split_list(self.root / "testing_list.txt")

        self.samples: list[tuple[Path, int]] = self._build_index()
        self._noise_clips: list[np.ndarray] | None = None  # lazy-loaded

    # -- index building -----------------------------------------------------

    def _in_split(self, rel_key: str) -> bool:
        """Does this 'word/file.wav' key belong to the requested split?"""
        if rel_key in self.test_list:
            return self.split == "test"
        if rel_key in self.val_list:
            return self.split == "val"
        # everything not listed is training data
        return self.split == "train"

    def _build_index(self) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []

        # 1) positives ------------------------------------------------------
        if self.positive_dir is not None:
            # real recorded wake words: every clip in the folder is positive
            for wav in sorted(self.positive_dir.rglob("*.wav")):
                samples.append((wav, 1))
        else:
            # temporary: use chosen Speech Commands word(s) as positives
            for word in sorted(self.positive_words):
                word_dir = self.root / word
                if not word_dir.is_dir():
                    continue
                for wav in sorted(word_dir.glob("*.wav")):
                    rel_key = f"{word}/{wav.name}"
                    if self._in_split(rel_key):
                        samples.append((wav, 1))

        # 2) negatives: all other word folders ------------------------------
        skip = self.positive_words | {"_background_noise_"}
        for word_dir in sorted(self.root.iterdir()):
            if not word_dir.is_dir() or word_dir.name in skip:
                continue
            for wav in sorted(word_dir.glob("*.wav")):
                rel_key = f"{word_dir.name}/{wav.name}"
                if self._in_split(rel_key):
                    samples.append((wav, 0))

        if not samples:
            raise RuntimeError(
                f"No samples found for split={self.split!r} under {self.root}. "
                "Check that download_data.py ran and the path is correct."
            )
        return samples

    # -- noise source (for add_noise augmentation) --------------------------

    def _load_noise_clips(self) -> list[np.ndarray]:
        if self._noise_clips is not None:
            return self._noise_clips
        clips = []
        noise_dir = self.root / "_background_noise_"
        if noise_dir.is_dir():
            for wav in sorted(noise_dir.glob("*.wav")):
                # load full noise clip (don't crop to 1 s: it's a long source)
                data, sr = _load_raw(wav)
                clips.append(data)
        self._noise_clips = clips
        return clips

    # -- PyTorch protocol ---------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        wav = load_audio(
            str(path),
            target_sr=SAMPLE_RATE,
            mono=True,
            target_length_s=CLIP_SECONDS,
        ).astype(np.float32)

        # ---- waveform-level augmentation (train only) --------------------
        if self.training:
            if self.use_time_shift:
                wav = augment.random_time_shift(wav, self.max_shift_samples)
            if self.use_add_noise:
                noise_clips = self._load_noise_clips()
                if noise_clips:
                    noise = self._rng.choice(noise_clips)
                    snr = self._rng.uniform(*self.snr_db_range)
                    wav = augment.add_noise(wav, noise, snr_db=snr)

        # ---- features: log-mel (40, 98) ----------------------------------
        log_mel = features.log_mel_spectrogram(
            wav, sr=SAMPLE_RATE, n_fft=N_FFT,
            hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_mels=N_MELS,
        ).astype(np.float32)

        # ---- spectrogram-level augmentation (train only) -----------------
        if self.training and self.use_spec_augment:
            log_mel = augment.spec_augment(log_mel)

        # (40, 98) -> (1, 40, 98): add channel dim for CNN
        feat = torch.from_numpy(log_mel).unsqueeze(0)
        return feat, torch.tensor(label, dtype=torch.long)


def _load_raw(path: Path) -> tuple[np.ndarray, int]:
    """Load a wav at native rate/length as float32 mono (for noise clips)."""
    import soundfile as sf
    data, sr = sf.read(str(path))
    data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, sr


if __name__ == "__main__":
    # Smoke test — requires data/raw/speech_commands to exist.
    train = KeywordSpottingDataset(split="train", seed=0)
    val = KeywordSpottingDataset(split="val", seed=0)
    test = KeywordSpottingDataset(split="test", seed=0)

    print(f"train / val / test sizes: {len(train)} / {len(val)} / {len(test)}")

    n_pos = sum(lbl for _, lbl in train.samples)
    print(f"train positives / total: {n_pos} / {len(train)}")

    feat, label = train[0]
    print("feature shape:", feat.shape, "dtype:", feat.dtype)   # (1, 40, 98) float32
    print("label:", label.item())

    # val must NOT be augmented
    assert val.training is False
    print("val.training =", val.training, "(should be False)")