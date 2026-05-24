"""Audio I/O file.

Make Audio I/O in one place so we can modify their attributes easier.
"""

import numpy as np
import soundfile as sf #read sound file
import torch #Tensor wave
import torchaudio.functional as AF #Resample


def load_audio(
        path,
        target_sr,
        mono,
        target_length_s
):
    data, sr = sf.read(path)
    print(f"shape={data.shape}, sr={sr}, dtype={data.dtype}")

    #convert to float32
    data = data.astype(np.float32)

    #check if mono
    if mono and data.ndim == 2:
        data = data.mean(axis=1)

    #resample (if sr != target_sr)
    if sr != target_sr:
        waveform = torch.from_numpy(data).unsqueeze(0) #shape: (1, n_samples)
        waveform = AF.resample(waveform, sr, target_sr)
        data = waveform.squeeze(0).numpy() #back to (n_samples,)
    print(f"after resample: shape={data.shape}, sr={target_sr}")

    #pad / crop
    target_n = int(target_sr * target_length_s)
    if len(data) > target_n:
        data = data[:target_n]
    elif len(data) < target_n:
        data = np.pad(data, (0, target_n - len(data)), mode='constant')

    return data


#Root mean square
def rms(x : np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))

#Signal Noise Ratio SNR
def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    #formula: 20 * log10(rms_signal / rms_noise)
    epsilon = 1e-10
    return 20 * np.log10(rms(signal) / (rms(noise) + epsilon))

if __name__ == "__main__":
    x = load_audio(r"D:\learnMLDSP\edgeaudio-ml\data\raw\esc50\audio\1-100032-A-0.wav", 16000, True, 1.0)
    assert len(x) == int(16000 * 1.0)

    # rms of silence == 0
    print(rms(np.zeros(1000)))                          # → 0.0

    # rms of unit-amplitude sine ≈ 1/sqrt(2) ≈ 0.707
    t = np.arange(16000) / 16000
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    print(rms(sine))                                    # → ~0.707

    # snr of signal vs itself → 0 dB (same amplitude)
    # snr of signal vs 0.1*signal → 20 dB (amplitude 10x → 20 log10(10) = 20)
    print(snr_db(sine, 0.1 * sine))                     # → 20.0
