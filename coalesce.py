import csv
import math
import os
from pathlib import Path
import librosa
import numpy as np
import json
from pydub import AudioSegment
from ast import literal_eval as make_tuple


def pad(arr):
    if arr.shape[1] > 1723:
        return arr[:, 0:1723]
    return np.repeat(arr, (math.ceil(1723 / arr.shape[1])), axis=1)[:, 0:1723]


file = open("./ECS/esc50.csv");
data = csv.reader(file, delimiter=",")
next(data);

x = np.zeros((0, 20, 431))
# y = np.zeros((0, 128, 1723))
y = np.zeros(0)

direction = "./ECS/audio/44100"
directory = Path(direction)

count = 0;

for row in data:
    audio, sampleRate = librosa.load(directory / row[0], sr=44100)

    # mfccs = librosa.feature.mfcc(y=audio, sr=sampleRate, n_mfcc=40, hop_length=128, n_fft=512)
    mfccs = librosa.feature.mfcc(y=audio, sr=sampleRate, n_mfcc=20)
    # mfccs = pad(mfccs)
    mfccs = mfccs.reshape((1, 20, 431))
    x = np.append(x, mfccs, axis=0)

    # spectrogram = librosa.feature.melspectrogram(y=audio, sr=sampleRate, hop_length=128, n_fft=512)
    # # spectrogram = pad(spectrogram)
    # spectrogram = spectrogram.reshape((1, 128, 1723))
    # y = np.append(y, spectrogram, axis=0)

    y = np.append(y, int(row[2]))
    count += 1
    print(count, end="\r")

file.close();

with open("vectorsXDefault.csv", "wb") as file:
    np.savetxt(file, x.reshape(x.shape[0], x.shape[1] * x.shape[2]), delimiter=",")

with open("vectorsYDefault.csv", "wb") as file:
    np.savetxt(file, y, delimiter=",")

# with open("vectorsZ.csv", "wb") as file:
#     np.savetxt(file, z.reshape(z.shape[0], z.shape[1] * z.shape[2]), delimiter=",")
