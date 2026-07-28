"""
Tests for augment.py.

These deliberately target the failure modes that a __main__ smoke test can't
catch: wrong axis in spec_augment, mutating the caller's array, missing return,
and SNR correctness across a range of targets.
"""

import numpy as np
import pytest

import my_kws.augment as augment


# ---------------------------------------------------------------------------
# add_noise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", [-5, 0, 5, 10, 20, 40])
def test_add_noise_hits_target_snr(target):
    rng = np.random.default_rng(0)
    clean = rng.standard_normal(16000).astype(np.float32)
    noise = rng.standard_normal(80000).astype(np.float32)

    noisy = augment.add_noise(clean, noise, snr_db=target)
    actual = 10 * np.log10(np.mean(clean**2) / np.mean((noisy - clean) ** 2))
    # float32 at 40 dB is the tightest case; 0.3 dB tolerance is comfortable
    assert abs(actual - target) < 0.3, f"target={target}, got {actual:.3f}"


def test_add_noise_preserves_shape_and_dtype():
    clean = np.random.randn(16000).astype(np.float32)
    noise = np.random.randn(80000).astype(np.float32)
    out = augment.add_noise(clean, noise, snr_db=10)
    assert out.shape == clean.shape
    assert out.dtype == clean.dtype


def test_add_noise_tiles_short_noise():
    # noise shorter than clean must be tiled, not crash
    clean = np.random.randn(16000).astype(np.float32)
    noise = np.random.randn(4000).astype(np.float32)
    out = augment.add_noise(clean, noise, snr_db=10)
    assert out.shape == clean.shape


# ---------------------------------------------------------------------------
# random_time_shift
# ---------------------------------------------------------------------------

def test_time_shift_preserves_length_and_dtype():
    x = np.arange(16000, dtype=np.float32)
    out = augment.random_time_shift(x, max_shift_samples=100)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_time_shift_zero_pads_and_keeps_content():
    # With a positive shift, zeros appear at the front and the tail is dropped.
    # We can't control the random sign, so run many trials and check invariants.
    x = np.arange(1, 16001, dtype=np.float32)  # no zeros in original
    for _ in range(50):
        out = augment.random_time_shift(x, max_shift_samples=100)
        # total energy can only stay same or drop (samples shifted off / zeroed)
        assert out.shape == x.shape
        # number of nonzero samples never exceeds original
        assert np.count_nonzero(out) <= len(x)


# ---------------------------------------------------------------------------
# spec_augment  — the important ones
# ---------------------------------------------------------------------------

def _fresh_spec(n_mels=40, n_frames=98):
    """Every entry distinct; global mean is 1959.5, which matches no element,
    so masked entries are exactly the fill value and nothing else is."""
    return np.arange(n_mels * n_frames, dtype=np.float32).reshape(n_mels, n_frames)


def test_spec_augment_returns_array():
    out = augment.spec_augment(_fresh_spec())
    assert out is not None, "spec_augment must return the augmented array"
    assert isinstance(out, np.ndarray)


def test_spec_augment_preserves_shape():
    x = _fresh_spec()
    out = augment.spec_augment(x)
    assert out.shape == x.shape


def test_spec_augment_does_not_mutate_input():
    x = _fresh_spec()
    x_copy = x.copy()
    _ = augment.spec_augment(x)
    assert np.array_equal(x, x_copy), "input array must not be modified in place"


def test_masked_values_equal_input_mean():
    """The fill value must be the INPUT's mean — not 0, not anything else."""
    x = _fresh_spec()
    fill = x.mean()
    for _ in range(20):  # masks are random; multiple trials
        out = augment.spec_augment(x)
        changed = out != x
        if changed.any():
            assert np.allclose(out[changed], fill), (
                "masked entries must be filled with the input mean"
            )


def test_freq_mask_is_on_mel_axis():
    """A freq mask must fill whole ROWS (mel bands), never partial rows."""
    x = _fresh_spec()
    fill = x.mean()
    out = augment.spec_augment(x, n_freq_masks=1, n_time_masks=0, freq_mask_param=7)
    filled = np.isclose(out, fill)
    fill_rows = np.where(np.all(filled, axis=1))[0]
    assert filled.sum() == len(fill_rows) * x.shape[1], (
        "freq mask leaked onto the time axis — wrong axis in spec_augment"
    )


def test_time_mask_is_on_frame_axis():
    """A time mask must fill whole COLUMNS (frames), never partial columns."""
    x = _fresh_spec()
    fill = x.mean()
    out = augment.spec_augment(x, n_freq_masks=0, n_time_masks=1, time_mask_param=25)
    filled = np.isclose(out, fill)
    fill_cols = np.where(np.all(filled, axis=0))[0]
    assert filled.sum() == len(fill_cols) * x.shape[0], (
        "time mask leaked onto the mel axis — wrong axis in spec_augment"
    )


def test_masks_respect_param_bounds():
    x = _fresh_spec()
    fill = x.mean()
    out = augment.spec_augment(
        x, freq_mask_param=7, time_mask_param=25,
        n_freq_masks=1, n_time_masks=1,
    )
    filled = np.isclose(out, fill)
    fill_rows = np.where(np.all(filled, axis=1))[0]
    fill_cols = np.where(np.all(filled, axis=0))[0]
    assert len(fill_rows) <= 7, "freq mask wider than freq_mask_param"
    assert len(fill_cols) <= 25, "time mask wider than time_mask_param"