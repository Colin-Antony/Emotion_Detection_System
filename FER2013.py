import os
import numpy as np
import pandas as pd
from keras.utils import to_categorical
import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras.regularizers import l1
from keras.optimizers import Adam

# Define paths for the train and test directories
train_dir = 'C:/Users/colin/Downloads/FER2013/train'
test_dir = 'C:/Users/colin/Downloads/FER2013/test'
# Define image dimensions
img_width, img_height = 48, 48

# Define batch size and number of epochs
batch_size = 64
epochs = 100

# Define the number of classes in the dataset
num_classes = 7

# Define the input shape of the images
input_shape = (img_width, img_height, 1)

# Create the data generators for train and test sets
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True,
                                   zoom_range=0.1)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    color_mode='grayscale',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    color_mode='grayscale',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Build the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])

early_stop = EarlyStopping(patience=10, monitor="val_loss", verbose=1)
# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size)

# Save the model
model.save('emotion.h5')

model.save_weights('emotion_weights.h5')



model_json = model.to_json()
with open('model_arch.json', 'w') as json_file:
    json_file.write(model_json)


# Evaluate the model on the test set
scores = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print("Test loss:", scores[0])
print("Test accuracy:", scores[1])






