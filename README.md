# Siren

A neural network that classifies common sounds for individuals who are hard of hearing.

[classify.py](classify.py) is used to run inference.

[train.py](train.py) is the file that was used for training the neural network (utilizing GPUs).

The [models](models) directory stores the saved TensorFlow models.

The [Flask](Flask) directory is used for the implementation of the neural network with web services.

## APIs + Libraries Utilized

The [TensorFlow](https://www.tensorflow.org/) deep learning library was used to create the neural networks using Convolutional Neural Networks with Spectrogram Analysis.

The [Librosa](https://pypi.org/project/librosa/) and [PyDub](https://pypi.org/project/pydub/) libraries were used for handling audio files with python.

The model was trained on the [ECS-50](https://github.com/karolpiczak/ESC-50) dataset.
