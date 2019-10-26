import os
import matplotlib.mlab as mlab
import librosa
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

direction = "./ECS/audio/44100"
directory = Path(direction)
audio, sampleRate = librosa.load(directory / '1-137-A-32.wav', sr=44100)
fig, ax = plt.subplots()
S, freqs, times, im = ax.specgram(audio, NFFT=4096, Fs=sampleRate,
                                  window=mlab.window_hanning,
                                  noverlap=4096 // 2)
S = S[::4, ::4]
print(np.shape(S))


def add(a, b):
    return a + b
