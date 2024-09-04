import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Check if a GPU is available
if tf.test.is_gpu_available():
    # Set the GPU as the default device
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[1], True)

# Define a function to load and preprocess images from a directory
def load_and_preprocess_images(directory, image_size):
    images = []
    labels = []
    emotions_dict = {
        "angry": 0,
        "disgust": 1,
        "fear": 2,
        "happy": 3,
        "neutral": 4,
        "sad": 5,
        "surprise": 6
    }

    for emotion_folder in os.listdir(directory):
        emotion_label = emotions_dict[emotion_folder]
        emotion_path = os.path.join(directory, emotion_folder)
        
        for image_file in os.listdir(emotion_path):
            image_path = os.path.join(emotion_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, image_size)
            image = image / 255.0  # Normalize pixel values to the range [0, 1]
            images.append(image)
            labels.append(emotion_label)

    return np.array(images), tf.keras.utils.to_categorical(labels, num_classes=len(emotions_dict))

# Define the directory paths for training and validation data
train_directory = "C:/Users/Uday/Downloads/FaceRecognition/images/train"  # Update with the path to your training data folder
validation_directory = "C:/Users/Uday/Downloads/FaceRecognition/images/validation"  # Update with the path to your validation data folder

# Load and preprocess training and validation data
image_size = (48, 48)  # Adjust the image size based on your model's input shape
x_train, y_train = load_and_preprocess_images(train_directory, image_size)
x_val, y_val = load_and_preprocess_images(validation_directory, image_size)

# Define the CNN model
def create_emotion_model(input_shape, num_classes):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the output for the fully connected layers
    model.add(layers.Flatten())

    # Fully connected layers
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Define the input shape and number of classes (emotions)
input_shape = (48, 48, 1)  # Adjust the image size based on your model's input shape
num_classes = 7  # Number of emotion classes (e.g., happy, sad, angry, etc.)

# Create the model
emotion_model = create_emotion_model(input_shape, num_classes)

# Compile the model
emotion_model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

# Train the model
batch_size = 64
epochs = 20
history = emotion_model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_val, y_val))

# Evaluate the model on the test set
test_directory = "C:/Users/Uday/Downloads/FaceRecognition/images/train"  # Update with the path to your test data folder
x_test, y_test = load_and_preprocess_images(test_directory, image_size)
test_loss, test_acc = emotion_model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# Make predictions on the test set
y_pred = emotion_model.predict(x_test)

# Convert one-hot encoded predictions to class labels
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Generate a classification report and confusion matrix
print(classification_report(y_true_labels, y_pred_labels))
print(confusion_matrix(y_true_labels, y_pred_labels))
