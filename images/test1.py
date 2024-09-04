import cv2
import numpy as np
import tensorflow as tf

# Load the saved model
model_filename = "emotion_model.h5"  # Replace with the path to your saved model
emotion_model = tf.keras.models.load_model(model_filename)

# Define a function to preprocess an image
def preprocess_image(image_path, image_size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, image_size)
    image = image / 255.0  # Normalize pixel values to the range [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define a function to predict emotion from an image
def predict_emotion(image_path, model):
    image_size = (48, 48)  # Adjust the image size based on your model's input shape
    preprocessed_image = preprocess_image(image_path, image_size)
    emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    predictions = model.predict(preprocessed_image)
    predicted_emotion = emotion_labels[np.argmax(predictions)]
    return predicted_emotion

# Example usage:
image_path = "C:/Users/Uday/Downloads/FaceRecognition/images/train\disgust/2939.jpg"  # Replace with the path to your test image
predicted_emotion = predict_emotion(image_path, emotion_model)
print(f"Predicted Emotion: {predicted_emotion}")
