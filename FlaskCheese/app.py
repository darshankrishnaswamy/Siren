import tensorflow as tf

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras.models import load_model
# from tensorflow.keras.models import load_model
#
from flask import Flask, request, jsonify
import matplotlib.mlab as mlab

app = Flask(__name__)

import numpy as np
import json

model = load_model("../models/model3.hdf5", compile=False)
model._make_predict_function()

mean = 1.626437693371607e-06
stddev = 7.178861976170184e-05

sampleRate = 44100
audio = np.zeros(220500)

spectrogram, frequencies, times = mlab.specgram(audio, NFFT=4096, Fs=sampleRate,
                                                window=mlab.window_hanning,
                                                noverlap=4096 // 2)
spectrogram = spectrogram[::4, ::4]
spectrogram = spectrogram.reshape((513, 27))

spectrogram -= mean
spectrogram /= stddev
print(model.predict(spectrogram.reshape((1, 513, 27, 1))))


@app.route("/", methods=["POST"])
def spectrogram():
    try:
        audio = np.array(json.loads(request.data))
        sampleRate = 44100
        if audio.shape[0] != 220500:
            return jsonify(result=-1, status=1)
            # audio = np.zeros(220500)
        spectrogram, frequencies, times = mlab.specgram(audio, NFFT=4096, Fs=sampleRate,
                                                        window=mlab.window_hanning,
                                                        noverlap=4096 // 2)
        spectrogram = spectrogram[::4, ::4]
        spectrogram = spectrogram.reshape((513, 27))

        spectrogram -= mean
        spectrogram /= stddev
        result = np.argmax(model.predict(spectrogram.reshape((1, 513, 27, 1))))

        print(result)

        return jsonify(result=int(result), status=0)
    except Exception as e:
        print(e)
        return jsonify(result=-1, status=1)


app.run(host="0.0.0.0", port=5000, debug=True, threaded=False)
