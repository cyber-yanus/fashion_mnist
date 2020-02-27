from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt


PATH = 'dataset'

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_pikachus_dir = os.path.join(train_dir, 'pikachus')
train_clothes_dir = os.path.join(train_dir, 'clothes')

validation_pikachus_dir = os.path.join(validation_dir, 'pikachus')
validation_clothes_dir = os.path.join(validation_dir, 'clothes')


num_pikachus_tr = len(os.listdir(train_pikachus_dir))
num_clothes_tr = len(os.listdir(train_clothes_dir))

num_pikachus_val = len(os.listdir(validation_pikachus_dir))
num_clothes_val = len(os.listdir(validation_clothes_dir))


total_train = num_pikachus_tr + num_clothes_tr
total_val = num_pikachus_val + num_clothes_val


batch_size = 128
epochs = 15
IMG_SIZE = 150


train_image_generator = ImageDataGenerator(rescale = 1./255)
validation_image_generator = ImageDataGenerator(rescale = 1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size = batch_size,
                                                           directory = train_dir,
                                                           shuffle = True,
                                                           target_size = (IMG_SIZE, IMG_SIZE),
                                                           class_mode = 'binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_SIZE, IMG_SIZE),
                                                              class_mode='binary')


sample_training_images, _ = next(train_data_gen)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    

plotImages(sample_training_images[:5])



model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
    ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.summary()


history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val
)


model.save('pikachu_classif.h5')










    
