# Digit Recognition Project

## Overview
This project implements a digit recognition model using TensorFlow and Keras. The model is trained to recognize handwritten digits (0-9) from images. It utilizes a convolutional neural network (CNN) for classification.

## Files in this Repository
- `digit_recognition.py`: A script that loads the trained model and predicts the digit from a given image.
- `model_training.py`: A script to train the digit recognition model on the MNIST dataset.
- `digit_recognition_model.h5`: The trained model file that can be used for predictions.
- `requirements.txt`: A list of Python packages required to run the project.
- `README.md`: This file, which contains an overview and instructions.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/iqbalahad/digit_recognition.git
   cd digit_recognition
   ```
2. **Set Up a Virtual Environment**:
   
   It’s recommended to use a virtual environment to manage dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

3. **Install Dependencies**:

Install the required packages using requirements.txt:
```bash
pip install -r requirements.txt
```

## Usage

1. **Predicting a digit**


To predict a digit from an image, run the following command:

```bash
python digit_recognition.py
```
You will be prompted to enter the path to the image file.


2. **Training the Model**

To train the digit recognition model, run:

```bash
python model_training.py
```
This script will train the model on the MNIST dataset and save the trained model as digit_recognition_model.h5.

## Future Work
A GUI (Graphical User Interface) will be added to enhance user interaction with the digit recognition model.

