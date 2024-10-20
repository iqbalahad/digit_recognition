import tensorflow as tf 
from keras import Sequential 
from keras import layers as l

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"Training data: {x_train.shape}, Training labels: {y_train.shape}")
print(f"Test data: {x_test.shape}, Test labels: {y_test.shape}")

# Preprocess the data
x_train = x_train.astype("float32") / 255.0 
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)  
x_test = x_test.reshape(-1, 28, 28, 1)

# Build the model
model = Sequential([
    l.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    l.MaxPooling2D(pool_size=(2, 2)),
    l.Conv2D(64, (3, 3), activation='relu'),
    l.MaxPooling2D(pool_size=(2, 2)),
    l.Flatten(),
    l.Dense(128, activation='relu'),
    l.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=7, validation_data=(x_test, y_test))

# Save the model
model.save("digit_recognition_model.h5")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")
