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
usefulClasses = [0, 1, 2, 3, 5, 6, 12, 19, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
map = [0, 1, 2, 3, 0, 4, 5, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
useful = []
usefulC = []
noise = []
# y = np.zeros((0, 128, 1723))
y = np.zeros(0)

direction = "./ECS/audio/44100"
directory = Path(direction)

count = 0;

for row in data:
    # audio, sampleRate = librosa.load(directory / row[0], sr=44100)

    # mfccs = mfccs.reshape((1, 20, 431))
    # x = np.append(x, mfccs, axis=0)

    x = np.append(x, spectrogram, axis=0)
    if int(row[2]) in usefulClasses:
        useful.append(spectrogram)
        y = np.append(y, map[int(row[2])])
        usefulC = np.append(usefulC, map[int(row[2])])
    else:
        noise.append(spectrogram)
        y = np.append(y, 19)

    count += 1
    print(count, end="\r")

file.close();

for i in range(useful.size()):
    spectrogram = useful[i]
    randomNoises = np.random.choice(noise.size(), 29, replace=False)
    for i in randomNoises:
        x.append(np.add(spectrogram, noise[i]))
        y.append(usefulC[i])

with open("vectorsXDefault.csv", "wb") as file:
    np.savetxt(file, x.reshape(x.shape[0], x.shape[1] * x.shape[2]), delimiter=",")

with open("vectorsYDefault.csv", "wb") as file:
    np.savetxt(file, y, delimiter=",")

# with open("vectorsZ.csv", "wb") as file:
#     np.savetxt(file, z.reshape(z.shape[0], z.shape[1] * z.shape[2]), delimiter=",")
