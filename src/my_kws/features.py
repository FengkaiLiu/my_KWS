import numpy as np
import matplotlib.pyplot as plt

#MelSpectrogram from scratch

#Normal Setup:
#frame_length = 400 samples = 25ms @ 16kHz
#hop_length = 160 samples = 10ms @ 16kHz
#overlapping: (400-160)/400 = 60%

#Split a 1D waveform into overlapping frames.
def frame(waveform: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    T = len(waveform)
    n_frames = 1 + (T - frame_length) // hop_length

    frames = np.zeros((n_frames, frame_length), dtype = waveform.dtype)
    for i in range(n_frames):
        start = i*hop_length
        end = i*hop_length + frame_length
        frames[i] = waveform[start : end]
    return frames

#Window function Hanning window
def window(frames: np.ndarray, window_type: str = "hann") -> np.ndarray:
    N = frames.shape[1]
    if window_type == "hann":
        n = np.arange(N)
        w = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
        w = w.astype(frames.dtype)
    else:
        raise ValueError(f"Unsupported window type: {window_type!r}")
    
    return frames * w
    
#Short time fourier transform
def stft(waveform: np.ndarray, n_fft: int = 512, hop_length: int = 160, win_length: int = 400) -> np.ndarray:
    frames = frame(waveform, win_length, hop_length)

    frames = window(frames)

    padded = np.zeros((frames.shape[0], n_fft), dtype=frames.dtype)
    padded[:, :win_length] = frames

    spectrum = np.fft.rfft(padded, axis=1).astype(np.complex64)

    return spectrum

#Melspectrogram
def hz_to_mel(f):
    mel = 2595 * np.log10(1 + f / 700)
    return mel
    
def mel_to_hz(m):
    hz = 700 * (10**(m / 2595) - 1)
    return hz

def mel_filterbank(sr: int, n_fft: int, n_mels: int, f_min: float = 0.0, f_max: float | None = None) -> np.ndarray:
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sr/2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype = np.float32)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)

    #mel -> Hz -> FFT
    hz_points = mel_to_hz(mel_points)
    bins = hz_points * n_fft / sr

    #for each triangle i
    for i in range (n_mels):
        f_left = bins[i]
        f_center = bins[i+1]
        f_right = bins[i+2]
    
        #for every FFT bin k, fb[i, k]
        for k in range(n_fft // 2+1):
            if k < f_left:
                fb[i, k] = 0.0
            elif k < f_center:
                fb[i, k] = (k - f_left) / (f_center - f_left)
            elif k < f_right:
                fb[i, k] = (f_right - k) / (f_right - f_center)
            else:
                fb[i, k] = 0.0

    return fb

def log_mel_spectrogram(
        waveform: np.ndarray,
        sr: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 40,
        eps: float = 1e-10,
) -> np.ndarray:
    thestft = stft(waveform, n_fft, hop_length, win_length)
    power_spec = np.abs(thestft)**2
    mel_fb = mel_filterbank(sr, n_fft, n_mels)
    mel_spec = power_spec @ mel_fb.T
    log_mel = np.log(mel_spec + eps)

    return log_mel

if __name__ == "__main__":
    x = np.arange(16000, dtype=np.float32)
    f = frame(x, 400, 160)
    print(f.shape)      #Expected (98, 400)
    print(f[0, :3])     #Expected [0. 1. 2.]
    print(f[1, :3])     #Expected [160. 161. 162.]
    print(f[-1, -1])    #Expected 15919

    # 0 by default
    N = 400
    n = np.arange(N)
    w = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    print(w[0], w[-1], w[(N-1)//2])      # → 0.0, 0.0, ≈1.0

    # windowing
    ones = np.ones((1, 400), dtype=np.float32)
    wf = window(ones)
    print(wf[0, 0], wf[0, -1])           # → 0.0, 0.0
    print(wf[0, 199], wf[0, 200])        # → ≈1.0, ≈1.0 (中点附近)

    # dtype 
    print(wf.dtype)                       # → float32 ✓ (修之前会是 float64)

    #stft
    S = stft(np.arange(16000, dtype=np.float32))
    print(S.shape) # Expected (98, 257)
    print(S.dtype) # Expected complex64/complex128

    print(hz_to_mel(0))
    print(mel_to_hz(hz_to_mel(1000)))

    fb = mel_filterbank(16000, 512, 40)
    plt.plot(fb.T)
    plt.show()

    #mel-spectrogram (Shape Inspect)
    x = np.random.randn(16000).astype(np.float32)
    log_mel = log_mel_spectrogram(x)
    print(log_mel.shape) # (98, 40)
    print(log_mel.dtype) # float32

    #signal visualization
    t = np.arange(16000) / 16000
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    log_mel = log_mel_spectrogram(sine)
    plt.imshow(log_mel.T, origin='lower', aspect='auto')
    plt.xlabel('frame') 
    plt.ylabel('mel bin')
    plt.colorbar()
    plt.show()

