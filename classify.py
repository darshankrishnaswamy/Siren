import tensorflow as tf

from keras.models import load_model
# from tensorflow.keras.models import load_model
import matplotlib.mlab as mlab
import librosa

import numpy as np

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

model = load_model("./models/model3.hdf5", compile=False)
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
model.predict(spectrogram.reshape((1, 513, 27, 1)))

map = ["Dog", "Rooster", "Pig", "Cow", "Cat", "Hen", "Fire", "Thunderstorm", "Glass Breaking", "Helicopter", "Chainsaw",
       "Siren", "Car Horn", "Engine", "Train", "Church Bells", "Airplane", "Fireworks", "Handsaw", "Nothing"]
usefulClasses = [0, 1, 2, 3, 5, 6, 12, 19, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
map2 = [0, 1, 2, 3, 0, 4, 5, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0,
        0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

try:
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    while 1:
        filename = input("\nFile?\n")
        audio, sampleRate = librosa.load("./ECS/audio/44100/" + filename, sr=44100)
        spectrogram, frequencies, times = mlab.specgram(audio, NFFT=4096, Fs=sampleRate,
                                                        window=mlab.window_hanning,
                                                        noverlap=4096 // 2)
        spectrogram = spectrogram[::4, ::4]
        spectrogram = spectrogram.reshape((513, 27))

        spectrogram -= mean
        spectrogram /= stddev
        prediction = np.argmax(model.predict(spectrogram.reshape((1, 513, 27, 1))))
        # print(prediction)
        print("\nPrediction: " + map[int(prediction)])
        actual = int(filename[filename.rfind("-") + 1: -4]);
        if actual in usefulClasses:
            print("Actual: " + map[map2[actual]]+"\n")
        else:
            print("Actual: " + "Nothing" + "\n")
except Exception as e:
    print(e)
    pass
