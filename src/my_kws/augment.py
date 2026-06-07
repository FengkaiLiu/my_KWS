import numpy as np

def add_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float, eps: float = 1e-12) -> np.ndarray:
    if len(noise) > len(clean):
        start = np.random.randint(0, len(noise) - len(clean) + 1)
        noise = noise[start : start + len(clean)]
    elif len(noise) < len(clean):
        n_repeats = (len(clean) + len(noise) - 1) // len(noise)
        noise = np.tile(noise, n_repeats)[:len(clean)]
    p_signal = np.mean(clean ** 2)
    p_noise = np.mean(noise ** 2)
    alpha = np.sqrt(p_signal / ((p_noise + 1e-12) * 10 ** (snr_db / 10)))
    return (clean + alpha * noise).astype(clean.dtype)

def random_time_shift(waveform: np.ndarray, max_shift_samples: int) -> np.ndarray:
    shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)

    if shift > 0:
        shifted = np.concatenate([np.zeros(shift, dtype=waveform.dtype), waveform[:-shift]])
    elif shift < 0:
        shifted = np.concatenate([waveform[-shift:], np.zeros(-shift, dtype=waveform.dtype)])
    else:
        shifted = waveform 

    return shifted

def spec_augment(
        log_mel: np.ndarray,
        time_mask_param: int = 25,
        freq_mask_param: int = 7,
        n_time_masks: int = 1,
        n_freq_masks: int = 1,
) -> np.ndarray:
    out = log_mel.copy()
    n_frames = out.shape
    n_mels = out.shape

    # freq mask
    for _ in range(n_freq_masks):
        

    
    # time mask
    for _ in range(n_time_masks):




if __name__ == "__main__":
    clean = np.random.randn(16000).astype(np.float32)
    noise = np.random.randn(80000).astype(np.float32)   # 5x length

    noisy = add_noise(clean, noise, snr_db=10)
    print(noisy.shape, noisy.dtype)   # (16000,) float32

    # SNR ≈ 10 dB
    actual_noise = noisy - clean
    actual_snr = 10 * np.log10(np.mean(clean**2) / np.mean(actual_noise**2))
    print(f"target=10dB, actual={actual_snr:.2f}dB")   # 应该 ≈ 10.00

    # SNR edge test
    for target in [-5, 0, 5, 10, 20, 40]:
        noisy = add_noise(clean, noise, snr_db=target)
        actual = 10 * np.log10(np.mean(clean**2) / np.mean((noisy - clean)**2))
        print(f"target={target:+3d}dB, actual={actual:+.2f}dB")

    # RTS
    x = np.arange(16000, dtype=np.float32)
    shifted = random_time_shift(x, max_shift_samples=100)
    print(shifted.shape, shifted.dtype)   # (16000,) float32
    print(shifted[:5], shifted[-5:])       # 0, Original value