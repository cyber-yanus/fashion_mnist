from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow и tf.keras
import tensorflow as tf
from tensorflow import keras

# Вспомогательные библиотеки
import math
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels


with open('train-images-idx3-ubyte.gz','rb') as f:
    train_images = extract_images(f)

with open('train-labels-idx1-ubyte.gz','rb') as f:
    train_labels = extract_labels(f)

with open('t10k-images-idx3-ubyte.gz','rb') as f:
    test_images = extract_images(f)

with open('t10k-labels-idx1-ubyte.gz','rb') as f:
    test_labels = extract_labels(f)


train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = np.squeeze(train_images, 3)
test_images = np.squeeze(test_images, 3)


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = keras.models.load_model('model.h5')
model.summary()


predictions = model.predict(test_images)

print(predictions[0])

print('предположение сети '+class_names[np.argmax(predictions[1])])
print('правильный ответ '+ class_names[test_labels[1]])
