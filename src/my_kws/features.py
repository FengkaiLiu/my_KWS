import numpy as np


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
    
def stft(waveform: np.ndarray, n_fft: int = 512, hop_length: int = 160, win_length: int = 400) -> np.ndarray:
    
    
    

if __name__ == "__main__":
    x = np.arange(16000, dtype=np.float32)
    f = frame(x, 400, 160)
    print(f.shape)      #Expected (98, 400)
    print(f[0, :3])     #Expected [0. 1. 2.]
    print(f[1, :3])     #Expected [160. 161. 162.]
    print(f[-1, -1])    #Expected 15919

    # 窗本身两端为 0
    N = 400
    n = np.arange(N)
    w = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    print(w[0], w[-1], w[(N-1)//2])      # → 0.0, 0.0, ≈1.0

    # 加窗后两端衰减到 0
    ones = np.ones((1, 400), dtype=np.float32)
    wf = window(ones)
    print(wf[0, 0], wf[0, -1])           # → 0.0, 0.0
    print(wf[0, 199], wf[0, 200])        # → ≈1.0, ≈1.0 (中点附近)

    # dtype 验证（修完细节 1 后）
    print(wf.dtype)                       # → float32 ✓ (修之前会是 float64)


