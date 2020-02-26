
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

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
     keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, steps_per_epoch=100)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

model.save('model.h5')

print('модель сохранена')
print('\nТочность на проверочных данных:', test_acc)
