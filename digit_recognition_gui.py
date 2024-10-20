import tensorflow as tf
from keras import models as m
import cv2
import numpy as np

# Load the trained model
model = m.load_model("digit_recognition_model.h5")

# Compile the model with an optimizer, loss function, and evaluation metric
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def preprocess_image(file_path):
    # Read the image in grayscale mode
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    # Resize the image to 28x28 pixels (standard for digit recognition)
    img = cv2.resize(img, (28, 28))
    # Normalize pixel values to the range [0, 1]
    img = img / 255.0
    # Add batch size and channel dimensions to the image array
    img_array = np.expand_dims(img, axis=(0, -1))
    return img_array

def predict_digit(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)
    
    # Check if the image was successfully processed
    if img_array is None:
        print("The image could not be processed.")  
        return  

    # Predict the digit using the model
    prediction = model.predict(img_array)
    print(f"Prediction: {prediction}")  
    # Find the digit with the highest predicted probability
    predicted_digit = np.argmax(prediction)
    print(f"The recognized digit is: {predicted_digit}")

if __name__ == "__main__":
    # Ask the user to input the image file path
    image_path = input("Please enter the path to the image file: ")  
    if image_path:
        predict_digit(image_path)
    else:
        print("No valid image path provided.")
