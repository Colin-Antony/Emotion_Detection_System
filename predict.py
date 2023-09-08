import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json, Sequential

# Load the model architecture from the JSON file
with open('my_model.json', 'r') as json_file:
    model_json = json_file.read()
loaded_model = model_from_json(model_json)

# Load the trained weights from the H5 file
loaded_model.load_weights('emotion_detection_model.h5')

test_dir = 'C:/Users/colin/Downloads/FER2013/test'
rescale_data = ImageDataGenerator(rescale=1. / 255)

test_data = rescale_data.flow_from_directory(directory=test_dir, target_size=(48, 48), color_mode='grayscale',
                                             batch_size=64,
                                             class_mode='categorical')

y_pred = loaded_model.predict(test_data)
y_predlabels = y_pred.argmax(axis=1)
print(y_predlabels)
