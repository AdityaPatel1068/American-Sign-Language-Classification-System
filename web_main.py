import streamlit as st
import cv2
import numpy as np
import time
from gtts import gTTS
import random
import tensorflow as tf

# Load the saved model (make sure the path is correct)
model_path = 'gesture_model.h5'  # Update this path if needed
model = tf.keras.models.load_model(model_path)

# Function to preprocess the image before passing it to the model
def preprocess_frame(frame):
    # Convert the frame to grayscale (if needed)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Convert grayscale to RGB if the model expects RGB input
    gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    
    # Resize the frame to match the input shape of your model
    resized_frame = cv2.resize(gray_frame, (256, 256))  # Adjust size as per your model input
    resized_frame = np.expand_dims(resized_frame, axis=0)  # Add batch dimension
    
    # Normalize the image (assuming your model expects values between 0 and 1)
    resized_frame = resized_frame / 255.0
    
    return resized_frame

# Function to predict the label from the preprocessed frame
def predict_gesture(frame):
    preprocessed_frame = preprocess_frame(frame)
    
    # Predict using the model
    prediction = model.predict(preprocessed_frame)
    
    # Assuming the model has multiple classes, get the predicted class
    predicted_class_index = np.argmax(prediction)
    
    # You can map the predicted index to a class label (replace 'class_labels' with your actual class labels)
    class_labels = ['hello', 'world', 'sign', 'language', 'test']  # Update this with your actual labels
    predicted_class_label = class_labels[predicted_class_index]
    
    return predicted_class_label

# Function to convert text to speech using gTTS
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    
    # Save the audio to a temporary file
    audio_path = "/tmp/temp_audio.mp3"  # Adjust this if running on Windows or another system
    tts.save(audio_path)
    
    return audio_path

# Initialize session state variables
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'predicted_sentence' not in st.session_state:
    st.session_state.predicted_sentence = ""
if 'streaming' not in st.session_state:
    st.session_state.streaming = False  # Stream state

# Function to navigate pages
def navigate_to(page):
    st.session_state.page = page

# Page 1: User Input
def user_input_page():
    st.title("Welcome to Ashtavakra")
    st.subheader("An interpreter for the deaf and hard of hearing.")
    st.write("Please enter your name to get started.")
    
    name = st.text_input("Your Name", placeholder="Enter your name here", key="name_input")
    
    if st.button("Next"):
        if name.strip():
            st.session_state.username = name
            st.session_state.page = 'guide_page'  # Navigate to guide page
        else:
            st.warning("Name cannot be empty. Please enter your name.")

# Page 2: Permissions Guide
def guide_page():
    st.title("Permissions Required")
    st.subheader(f"Hello, {st.session_state.username}! Please grant necessary permissions.")
    st.write("""
    To use this app, we need the following permissions:
    - **Camera:** To capture your gestures or sign language in real-time.
    - **Microphone:** To capture audio input for interpretation.
    """)
    st.write("""
    Please allow camera and microphone permissions when prompted by your browser.
    - Your camera will be used to capture video input.
    """)

    st.camera_input("Camera Permission Test", key="camera_test_placeholder")

    if st.button("Grant Permissions"):
        st.session_state.page = 'live_stream'

# Page 3: Live Video Stream and Prediction
def live_stream_page():
    st.title("Live Video Interpretation")
    st.write("The app will interpret your gestures/sign language and display sentences in English and Español.")

    # Video capture
    cap = cv2.VideoCapture(0)  # Use webcam
    stframe = st.empty()  # Streamlit frame for displaying video

    # Start and Stop Buttons
    if st.button("Start Stream"):
        st.session_state.streaming = True
    if st.button("Stop Stream"):
        st.session_state.streaming = False

    predicted_words = []

    while st.session_state.streaming and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera.")
            break

        # Show live video stream
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        stframe.image(frame_rgb, channels="RGB", caption="Live Video Stream", use_container_width=True)

        # Make a prediction using the loaded model
        predicted_label = predict_gesture(frame)  # Predict label from the frame
        predicted_words.append(predicted_label)

        # If enough words are collected, generate the sentence
        if len(predicted_words) >= 5:  # Example threshold
            predicted_sentence = " ".join(predicted_words)
            predicted_words = []  # Reset words buffer
            
            # Convert predicted sentence to speech
            audio_path = text_to_speech(predicted_sentence)
            st.session_state.predicted_sentence = predicted_sentence
            
            # Display the predicted sentence and provide the audio feedback
            st.success(f"Predicted Sentence: {predicted_sentence}")
            st.audio(audio_path, format="audio/mp3")

        # Delay for real-time streaming
        time.sleep(0.1)

    cap.release()
    stframe.image(np.zeros((480, 640, 3)), caption="Stream Stopped", use_container_width=True)  # Placeholder when stopped

# Render the appropriate page
if st.session_state.page == 'home':
    user_input_page()
elif st.session_state.page == 'guide_page':
    guide_page()
elif st.session_state.page == 'live_stream':
    live_stream_page()
