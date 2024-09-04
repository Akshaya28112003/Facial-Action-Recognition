import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
import keyboard  # Import the keyboard module

# Load the saved model
model_filename = "emotion_model.h5"  # Replace with the path to your saved model
emotion_model = tf.keras.models.load_model(model_filename)

# Load the face detection model from OpenCV (you can change the path accordingly)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a function to preprocess an image
def preprocess_image(image, image_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    image = cv2.resize(image, image_size)
    image = image / 255.0  # Normalize pixel values to the range [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define a function to predict emotion from an image
def predict_emotion(image, model):
    image_size = (48, 48)  # Adjust the image size based on your model's input shape
    preprocessed_image = preprocess_image(image, image_size)
    emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    predictions = model.predict(preprocessed_image)
    predicted_emotion = emotion_labels[np.argmax(predictions)]
    return predicted_emotion, predictions[0]

# Open a connection to the webcam (you may need to change the index)
cap = cv2.VideoCapture(0)

# Initialize variables for emotion tracking
frame_buffer = deque(maxlen=10)  # Buffer to store the last 10 predicted emotions
emotion_history = np.zeros((10, 7))  # Buffer to store the last 10 emotion probabilities
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop the face region from the frame
        face_roi = frame[y:y+h, x:x+w]

        # Predict emotion for the face region
        predicted_emotion, emotion_probabilities = predict_emotion(face_roi, emotion_model)

        # Store the predicted emotion and probabilities in the buffer
        frame_buffer.append(predicted_emotion)
        emotion_history = np.vstack((emotion_history[1:], emotion_probabilities))

        # Calculate the average emotion over the last frames
        avg_emotion = max(set(frame_buffer), key=frame_buffer.count)

        # Display the predicted emotion on the frame
        cv2.putText(frame, f"Emotion: {avg_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Create a graph in the corner to indicate emotions
    plt.figure(figsize=(4, 3))
    for i, emotion in enumerate(emotion_labels):
        plt.plot(emotion_history[:, i], label=emotion)
    plt.legend(loc='upper right')
    plt.title('Emotion Probabilities')
    plt.ylim(0, 1)
    plt.savefig('emotion_graph.png')
    plt.close()

    # Read the graph image and overlay it on the frame
    graph_image = cv2.imread('emotion_graph.png')
    frame[10:10 + graph_image.shape[0], -10 - graph_image.shape[1]:-10] = graph_image

    # Convert the BGR frame to RGB for Matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame with the predicted emotions and face rectangles using Matplotlib
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.1)
    plt.clf()

    # Check for 'q' key press to exit the loop
    if keyboard.is_pressed('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
