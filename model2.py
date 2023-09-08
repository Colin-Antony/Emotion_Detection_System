from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import ReduceLROnPlateau
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

# Perform Data Augmentation
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True,
                                   zoom_range=0.1)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Generate the training and testing data
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
model.add(
    Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(img_width, img_height, 1)))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(units=128, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.25))
model.add(Dense(units=64, activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(units=32, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(7, activation='softmax'))

# Compile the model
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])

print(model.summary())

# Train the model
print("Fitting the model..")
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=[lr_schedule])

# Save the model : the model, weights of the model, structure of the model
model.save('emotion.h5')

model.save_weights('emotion_weights.h5')

model_json = model.to_json()
with open('model_arch.json', 'w') as json_file:
    json_file.write(model_json)

# Evaluate the model on the test set
scores = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print("Test loss:", scores[0])
print("Test accuracy:", scores[1])
