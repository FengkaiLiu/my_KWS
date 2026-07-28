"""Dataset that reads precomputed log-mel .npy features instead of wavs.

Same semantics as KeywordSpottingDataset:
    - official validation_list.txt / testing_list.txt splits
      (train = everything else), so speaker assignment is identical
    - binary label: 1 if the word directory is in positive_words
    - output: ((1, 40, 98) float32 tensor, int label)

Differences:
    - no waveform-level augmentation (time shift / add noise) — the whole
      point of the cache experiment
    - SpecAugment still applied when training=True (it works on features)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import spec_augment


class CachedKeywordSpottingDataset(Dataset):
    def __init__(
        self,
        cache_root: str | Path = "data/cache/logmel",
        data_root: str | Path = "data/raw/speech_commands",
        split: str = "train",
        positive_words: tuple[str, ...] = ("yes",),
        training: bool = False,
        use_spec_augment: bool = True,
    ) -> None:
        cache_root = Path(cache_root)
        data_root = Path(data_root)
        if not cache_root.exists():
            raise FileNotFoundError(
                f"feature cache not found at {cache_root} — "
                "run `uv run python -m my_kws.precompute_features` first"
            )

        val_list = self._read_list(data_root / "validation_list.txt")
        test_list = self._read_list(data_root / "testing_list.txt")

        self.items: list[tuple[Path, int]] = []
        for d in sorted(cache_root.iterdir()):
            if not d.is_dir():
                continue
            label = 1 if d.name in positive_words else 0
            for f in sorted(d.glob("*.npy")):
                rel_wav = f"{d.name}/{f.stem}.wav"
                if rel_wav in val_list:
                    item_split = "val"
                elif rel_wav in test_list:
                    item_split = "test"
                else:
                    item_split = "train"
                if item_split == split:
                    self.items.append((f, label))

        if not self.items:
            raise RuntimeError(
                f"no items for split={split!r} — check cache_root and split name"
            )
        n_pos = sum(lbl for _, lbl in self.items)
        if n_pos == 0:
            raise RuntimeError(
                f"0 positive samples for positive_words={positive_words} — "
                "check the word directories exist in the cache"
            )

        self.training = training
        self.use_spec_augment = use_spec_augment

    @staticmethod
    def _read_list(path: Path) -> set[str]:
        return {
            line.strip().replace("\\", "/")
            for line in path.read_text().splitlines()
            if line.strip()
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        feat = np.load(path)                      # (40, 98) float32
        if self.training and self.use_spec_augment:
            feat = spec_augment(feat)
        x = torch.from_numpy(np.ascontiguousarray(feat)).unsqueeze(0)  # (1, 40, 98)
        return x.float(), label
