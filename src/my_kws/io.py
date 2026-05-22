"""Audio I/O file.

Make Audio I/O in one place so we can modify their attributes easier.
"""

import soundfile as sf #read sound file
import torch #Tensor wave
import torchaudio.functional as AF #Resample


def load_audio(
        path,
        target_sr,
        mono,
        target_length_s
)