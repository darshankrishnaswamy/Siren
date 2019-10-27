import code
import tensorflow as tf
from tensorflow.keras.models import load_model

name = input("Name:\n")

model = load_model("./models/" + str(name) + ".hdf5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("./models/" + str(name) + ".tflite", "wb") as file:
    file.write(tflite_model)

variables = globals().copy()
variables.update(locals())
shell = code.InteractiveConsole(variables)
shell.interact()
