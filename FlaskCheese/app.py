from flask import Flask, request, jsonify
import math
import matplotlib.mlab as mlab

app = Flask(__name__)

import librosa
import numpy as np

@app.route("/", methods=["POST"])
def specto():
    try:
        file = request.files['file']
        file.save("./audio.wav")
        audio, sampleRate = librosa.load("./audio.wav", sr=44100)
        spectrogram, freqs, times = mlab.specgram(audio, NFFT=4096, Fs=sampleRate,
                                              window=mlab.window_hanning,
                                              noverlap=4096 // 2)
        spectrogram = spectrogram[::4, ::4]
        spectrogram = spectrogram.reshape((513, 27))

        return jsonify(data=spectrogram.tolist())
    except:
        return jsonify(data="", status=1)
