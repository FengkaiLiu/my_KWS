import numpy as np




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



if __name__ == "__main__":
    x = np.arange(16000, dtype=np.float32)
    f = frame(x, 400, 160)
    print(f.shape)      #Expected (98, 400)
    print(f[0, :3])     #Expected [0. 1. 2.]
    print(f[1, :3])     #Expected [160. 161. 162.]
    print(f[-1, -1])    #Expected 15919


