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

    #resample (if sr != target_sr)
    if sr != target_sr:
        waveform = torch.from_numpy(data).unsqueeze(0) #shape: (1, n_samples)
        waveform = AF.resample(waveform, sr, target_sr)
        data = waveform.squeeze(0).numpy() #back to (n_samples,)

    print(f"after resample: shape={data.shape}, sr={target_sr}")
    
    return data

if __name__ == "__main__":
    x = load_audio(r"D:\Downloads\oboe_samples\oboe_samples\Ob-ord-A4-mf.wav", 16000, True, 1.0)
