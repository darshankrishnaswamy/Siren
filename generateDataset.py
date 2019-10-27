import csv
import math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
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


file = open("./ECS/esc50.csv")
data = csv.reader(file, delimiter=",")
next(data)

x = np.zeros((0, 513, 27))
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
fileCount = 0;

for row in data:
    audio, sampleRate = librosa.load(directory / row[0], sr=44100)
    print(audio.shape)

    spectrogram, freqs, times = mlab.specgram(audio, NFFT=4096, Fs=sampleRate,
                                              window=mlab.window_hanning,
                                              noverlap=4096 // 2)
    print(spectrogram.shape)
    break
    spectrogram = spectrogram[::4, ::4]
    spectrogram = spectrogram.reshape((1, 513, 27))

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
    if x.shape[0] >= 400:
        with open("./data/vectorsX" + str(fileCount) + ".csv", "wb") as file:
            np.savetxt(file, x.reshape(x.shape[0], x.shape[1] * x.shape[2]), delimiter=",")
        with open("./data/vectorsY" + str(fileCount) + ".csv", "wb") as file:
            np.savetxt(file, y, delimiter=",")
        x = np.zeros((0, 513, 27))
        y = np.zeros(0)
        fileCount += 1

file.close();

count = 0

for i in range(len(useful)):
    spectrogram = useful[i]
    randomNoises = np.random.choice(len(noise), 29, replace=False)
    for j in randomNoises:
        x = np.append(x, np.add(spectrogram, noise[j]), axis=0)
        y = np.append(y, usefulC[i])
        if x.shape[0] >= 400:
            with open("./data/vectorsX" + str(fileCount) + ".csv", "wb") as file:
                np.savetxt(file, x.reshape(x.shape[0], x.shape[1] * x.shape[2]), delimiter=",")
            with open("./data/vectorsY" + str(fileCount) + ".csv", "wb") as file:
                np.savetxt(file, y, delimiter=",")
            x = np.zeros((0, 513, 27))
            y = np.zeros(0)
            fileCount += 1
        count += 1
        print(str(count) + "       ", end="\r")

with open("vectorsX.csv", "wb") as file:
    np.savetxt(file, x.reshape(x.shape[0], x.shape[1] * x.shape[2]), delimiter=",")

with open("vectorsY.csv", "wb") as file:
    np.savetxt(file, y, delimiter=",")

# with open("vectorsZ.csv", "wb") as file:
#     np.savetxt(file, z.reshape(z.shape[0], z.shape[1] * z.shape[2]), delimiter=",")
