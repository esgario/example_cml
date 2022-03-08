import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

src_dataset = './dataset'

model = Sequential()

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))

print(model.summary())

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2))

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(model.summary())

train_datagen = ImageDataGenerator(
    brightness_range=[0.5, 1.5],
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator()

batch_size = 4

train_image_gen = train_datagen.flow_from_directory(
    os.path.join(src_dataset, 'train'),
    target_size=(64, 64),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_image_gen = valid_datagen.flow_from_directory(
    os.path.join(src_dataset, 'val'),
    target_size=(64, 64),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

results = model.fit_generator(
    train_image_gen,
    epochs=10,
    validation_data=val_image_gen
)

def plot_result(values, filename):
    "save plot result"
    plt.figure()
    plt.plot(values)
    plt.title(filename.rsplit(".", 1)[0])
    plt.savefig(filename)

plot_result(results.history['val_accuracy'], "val_accuracy.png")
plot_result(results.history['loss'], "loss.png")
