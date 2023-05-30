import os  # For interacting with the operating system
import cv2  # For computer vision tasks (for loading/processing images)
import numpy as np  # For numerical computing and array manipulation
import matplotlib.pyplot as plt  # For data visualization (for digits)
import tensorflow as tf  # For machine learning

import ssl  # Disable SSL to avoid problems
ssl._create_default_https_context = ssl._create_unverified_context

mnist = tf.keras.datasets.mnist

# Split the data into training data and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Scale down the data. Normalize the pixel values between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


############################# Creating the model #############################
 
# In order to change the model, uncomment the code then run the code again. 
# This code has been commented to avoid re making the model every time.

# # Create a basic sequential neural network
# model = tf.keras.models.Sequential()

# # Add a flatten layer, flatten 2D 28x28 into a 1D single layer
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# # Add dense layers, basic fully connected layers. Relu => f(x) = max(0, x)
# model.add(tf.keras.layers.Dense(128, activation='relu'))  # Hidden layer

# model.add(tf.keras.layers.Dense(128, activation='relu'))  # Hidden layer

# model.add(tf.keras.layers.Dense(10, activation='softmax'))  # Output layer => 0, 1, ..., 9 for digits. Softmax ensures all neurons add up to 1 (AKA confidence)

# # # Adjust the learning rate
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(x_train, y_train, epochs=40)

# # Save the model when it's done
# model.save('handwritten.model')

#####################################################################

# Load the model
model = tf.keras.models.load_model('handwritten.model')  # Load the model
loss, accuracy = model.evaluate(x_test, y_test)  # Test model accuracy
print(f'Loss: {loss * 100}%, Accuracy: {accuracy * 100}%')  # Print metrics

image_number = 0
while os.path.exists(f"handwriten_numbers/digit{image_number}.png"):  # Check if the file exists
    try:
        img = cv2.imread(f"handwriten_numbers/digit{image_number}.png")[:, :, 0]  # Read the file, we only care about shape
        img = np.invert(np.array([img]))
        prediction = model.predict(img)  # Get activation for all digit neurons
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error : only 28 x 28 images are supported.")
    finally:
        print(image_number)
        image_number += 1
