import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Define a function to load and preprocess images from a directory
def load_and_preprocess_images(directory, image_size):
    images = []
    labels = []
    emotions_dict = {
        "anger": 0,
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

# Define the directory path for the saved model
saved_model_path = "fine_tuned_emotion_model.h5"

# Load the pre-trained model
pretrained_model = load_model(saved_model_path)

# Define the directory path for additional data
additional_data_directory = "C:/Users/Uday/Downloads/archive (7)"

# Load and preprocess your additional data
image_size = (48, 48)
x_additional, y_additional = load_and_preprocess_images(additional_data_directory, image_size)

# Split the additional data into training and validation sets
x_train_additional, x_val_additional, y_train_additional, y_val_additional = train_test_split(
    x_additional, y_additional, test_size=0.2, random_state=42
)

# Fine-tune the model
fine_tuning_epochs = 10
fine_tuning_batch_size = 32

history_fine_tune = pretrained_model.fit(
    x_train_additional,
    y_train_additional,
    batch_size=fine_tuning_batch_size,
    epochs=fine_tuning_epochs,
    validation_data=(x_val_additional, y_val_additional)
)

# Save the fine-tuned model
fine_tuned_model_filename = "fine_tuned1.h5"
pretrained_model.save(fine_tuned_model_filename)
print(f"Fine-tuned model saved as {fine_tuned_model_filename}")
