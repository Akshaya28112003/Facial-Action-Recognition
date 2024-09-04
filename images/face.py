import cv2
import numpy as np
import tensorflow as tf

# Load the saved model
model_filename = "emotion_model.h5"  # Replace with the path to your saved model
emotion_model = tf.keras.models.load_model(model_filename)

# Define a function to preprocess an image
def preprocess_image(image, image_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, image_size)
    image = image / 255.0  # Normalize pixel values to the range [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define a function to predict emotion from a frame
def predict_emotion(frame, model):
    image_size = (48, 48)  # Adjust the image size based on your model's input shape
    preprocessed_image = preprocess_image(frame, image_size)
    emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    predictions = model.predict(preprocessed_image)
    predicted_emotion = emotion_labels[np.argmax(predictions)]
    return predicted_emotion

# Open the webcam
cap = cv2.VideoCapture(0)  # Use the default camera (you can change the index if you have multiple cameras)

while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    if not ret:
        break

    # Predict emotion and display it on the frame
    predicted_emotion = predict_emotion(frame, emotion_model)
    cv2.putText(frame, f"Emotion: {predicted_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
