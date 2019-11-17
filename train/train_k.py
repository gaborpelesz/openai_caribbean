#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os

from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, Reshape
from tensorflow.python.keras.layers import Dense
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential

# model = Sequential()

# model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))

# model.add(Dense(1024, activation="relu"))
# model.add(Dropout(0.4))
# model.add(Dense(512, activation="relu"))
# model.add(Dense(256, activation="relu"))
# model.add(Dropout(0.4))
# model.add(Dense(128, activation="relu"))
# model.add(Dense(64, activation="relu"))
# model.add(Dense(32, activation="relu"))
# model.add(Dense(5, activation = 'softmax'))

# model.layers[0].trainable = False

# model.summary()

# model.compile(optimizer='adam', metrics=['accuracy'],loss='categorical_crossentropy')

base_dir = 'stac/training_data'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

image_size = (224,224)
batch_size = 10
seed = 1

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', seed=seed)
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', seed=seed)

print(train_generator.class_indices)

# model.fit_generator(train_generator,validation_data=validation_generator,epochs=50)
# model.save('karibi.h5')
