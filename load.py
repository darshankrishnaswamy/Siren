import tensorflow as tf

# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

import os
import numpy as np

np.random.seed(42)  # re-seed generator

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

# imports from keras for neural net

from keras.layers import Dense, Input, Flatten, Dropout, LSTM
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Embedding
from keras.models import Model, Sequential
from keras import optimizers

from keras.models import load_model
from keras.models import model_from_json

from sklearn.model_selection import train_test_split

import collections, numpy
import sys
import re
import random
import keras

from keras import backend as K


def loadCsvSkipCommentAndRows(filename, skipRows):
    with open(filename) as f:
        lines = (line for line in f if not line.startswith('#'))
        return np.loadtxt(lines, delimiter=',', skiprows=skipRows)


def load():
    name = input("Name:\n")
    json_file = open('./models/' + str(name) + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights('./models/' + str(name) + '.h5')
    model.compile(loss="categorical_crossentropy", optimizer='adam')


def train():
    exec(open("train.py").read())


files = 5

images = np.zeros((0, 513, 27))
vectors = np.zeros(0)

for i in range(files+1):
    with open("./data/vectorsX"+str(i)+".csv", "rb") as file:
        imagesTemp = np.loadtxt(file, delimiter=',')
        images = np.append(images, imagesTemp.reshape((imagesTemp.shape[0], 513, 27)), axis=0)

    with open("./data/vectorsY"+str(i)+".csv", "rb") as file:
        vectors = np.append(vectors, np.loadtxt(file, delimiter=','))

combined = np.c_[images.reshape(len(images), -1), vectors.reshape(len(vectors), -1)]

np.random.shuffle(combined)

images = np.copy(combined[:, :images.size // len(images)]).reshape(images.shape)
vectors = np.copy(combined[:, images.size // len(images):]).reshape(vectors.shape)

# images -= images.min()
images -= images.mean()
images = images / np.std(images)

XT = images.reshape((images.shape[0], images.shape[1], images.shape[2], 1))
YT = to_categorical(vectors)

X, x, Y, y = train_test_split(XT, YT, test_size=0.2)

trainImmediately = input("Go to train file?\n")
if str(trainImmediately) != "n" and str(trainImmediately) != "N" and str(trainImmediately) != "":
    exec(open("train.py").read())
else:
    import code

    variables = globals().copy()
    variables.update(locals())
    shell = code.InteractiveConsole(variables)
    shell.interact()
