"""
Tests for datasets.py.

We don't need the real 100k-file dataset to test the logic that actually
breaks: split assignment, positive/negative labeling, and the "augment only
on train" rule. We build a tiny synthetic Speech-Commands-style folder with
real (silent) .wav files in a tmp dir.

Requires: torch, soundfile, and audio_io.py / features.py / augment.py present.
Run with: pytest test_datasets.py
"""

from pathlib import Path
import numpy as np
import soundfile as sf
import pytest

import my_kws.kws_datasets as kws_datasets
from my_kws.kws_datasets import KeywordSpottingDataset


def _write_wav(path: Path, seconds=1.0, sr=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (np.random.randn(int(sr * seconds)) * 0.01).astype(np.float32)
    sf.write(str(path), data, sr)


@pytest.fixture
def fake_root(tmp_path):
    """
    Build:
        yes/  (positive word)  -> 6 files
        no/   (negative)       -> 4 files
        up/   (negative)       -> 4 files
        _background_noise_/    -> 1 long noise file
        validation_list.txt, testing_list.txt
    """
    root = tmp_path / "speech_commands"
    files = {"yes": 6, "no": 4, "up": 4}
    all_keys = []
    for word, n in files.items():
        for i in range(n):
            name = f"clip_{i}.wav"
            _write_wav(root / word / name)
            all_keys.append(f"{word}/{name}")

    # long noise clip
    _write_wav(root / "_background_noise_" / "pink.wav", seconds=3.0)

    # put 1 file from each word into val, 1 into test (rest -> train)
    val = [k for k in all_keys if k.endswith("clip_0.wav")]
    test = [k for k in all_keys if k.endswith("clip_1.wav")]
    (root / "validation_list.txt").write_text("\n".join(val) + "\n")
    (root / "testing_list.txt").write_text("\n".join(test) + "\n")

    return root


def test_labels_positive_is_yes(fake_root):
    ds = KeywordSpottingDataset(root=fake_root, split="train",
                                positive_words=("yes",), seed=0)
    for path, label in ds.samples:
        expected = 1 if path.parent.name == "yes" else 0
        assert label == expected


def test_splits_are_disjoint_and_use_official_lists(fake_root):
    train = KeywordSpottingDataset(root=fake_root, split="train", seed=0)
    val = KeywordSpottingDataset(root=fake_root, split="val", seed=0)
    test = KeywordSpottingDataset(root=fake_root, split="test", seed=0)

    def keys(ds):
        return {f"{p.parent.name}/{p.name}" for p, _ in ds.samples}

    kt, kv, ke = keys(train), keys(val), keys(test)
    # no overlap between any pair
    assert kt.isdisjoint(kv)
    assert kt.isdisjoint(ke)
    assert kv.isdisjoint(ke)
    # val has exactly the clip_0 files, test the clip_1 files
    assert all(k.endswith("clip_0.wav") for k in kv)
    assert all(k.endswith("clip_1.wav") for k in ke)


def test_background_noise_never_a_sample(fake_root):
    for split in ("train", "val", "test"):
        ds = KeywordSpottingDataset(root=fake_root, split=split, seed=0)
        for path, _ in ds.samples:
            assert path.parent.name != "_background_noise_"


def test_training_flag_inferred_from_split(fake_root):
    assert KeywordSpottingDataset(root=fake_root, split="train").training is True
    assert KeywordSpottingDataset(root=fake_root, split="val").training is False
    assert KeywordSpottingDataset(root=fake_root, split="test").training is False


def test_getitem_shape_and_dtype(fake_root):
    import torch
    ds = KeywordSpottingDataset(root=fake_root, split="val", seed=0)  # no aug
    feat, label = ds[0]
    assert feat.shape == (1, kws_datasets.N_MELS, 98)   # (1, 40, 98)
    assert feat.dtype == torch.float32
    assert label.dtype == torch.long
    assert label.item() in (0, 1)


def test_val_getitem_is_deterministic(fake_root):
    # no augmentation on val => same item twice must be identical
    ds = KeywordSpottingDataset(root=fake_root, split="val", seed=0)
    f1, _ = ds[0]
    f2, _ = ds[0]
    import torch
    assert torch.equal(f1, f2)


def test_positive_dir_overrides_words(fake_root, tmp_path):
    # when positive_dir is given, every clip there is positive and
    # positive_words is ignored for positives
    recorded = tmp_path / "wakewords"
    _write_wav(recorded / "hey_computer_0.wav")
    _write_wav(recorded / "hey_computer_1.wav")

    ds = KeywordSpottingDataset(
        root=fake_root, split="train",
        positive_words=None, positive_dir=recorded, seed=0,
    )
    pos = [p for p, lbl in ds.samples if lbl == 1]
    assert len(pos) == 2
    assert all(p.parent.name == "wakewords" for p in pos)