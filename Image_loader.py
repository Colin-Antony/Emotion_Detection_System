from keras.models import load_model
import numpy as np
from PIL import Image

print("Starting")
# model path
model_path = 'emotion.h5'

model = load_model(model_path)

# image path
image_path = "DanielaSadcrop.jpg"

# Preprocessing
img = Image.open(image_path).convert('L') # grayscale conversion
img = img.resize((48, 48)) # match image dimenstions
img_arr = np.array(img).reshape((1, 48, 48, 1)) / 255. # reshape and normalize

# Make a prediction on the image
prediction = model.predict(img_arr)

# Print the predicted class
predicted_class = np.argmax(prediction)
print("Predicted class index:", predicted_class)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
print("Predicted class label:", class_labels[predicted_class])
