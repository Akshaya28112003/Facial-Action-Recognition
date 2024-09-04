import cv2
import numpy as np
import tensorflow as tf

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
    return predicted_emotion

# Open a connection to the webcam (you may need to change the index)
cap = cv2.VideoCapture(0)

# Initialize variables for emotion tracking and smoothing
frame_count = 0
smoothed_prediction = 4  # Initial prediction (Neutral)
alpha = 0.2  # Smoothing factor (adjust as needed)

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
        predicted_emotion = predict_emotion(face_roi, emotion_model)

        # Convert predicted_emotion to a numerical value (0-6)
        emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        predicted_emotion_value = emotion_labels.index(predicted_emotion)

        # Smooth the emotion prediction using EMA
        if frame_count == 0:
            smoothed_prediction = predicted_emotion_value
        else:
            smoothed_prediction = alpha * predicted_emotion_value + (1 - alpha) * smoothed_prediction

        # Convert the smoothed prediction back to an emotion label
        smoothed_emotion = emotion_labels[int(smoothed_prediction)]

        # Display the smoothed predicted emotion on the frame
        cv2.putText(frame, f"Emotion: {smoothed_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the frame with the predicted emotions and face rectangles
    cv2.imshow('Emotion Detection', frame)

    # Increment frame count
    frame_count += 1

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()