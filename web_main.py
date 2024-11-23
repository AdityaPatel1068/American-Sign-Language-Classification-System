import streamlit as st
import cv2
import numpy as np
import time
import openai
from openai import OpenAI

from deep_translator import GoogleTranslator
from dotenv import load_dotenv
import os
import random

# Load environment variables from .env file
load_dotenv()


# Mock word predictions for testing
def mock_word_predictions():
    words = ["hello", "world", "sign", "language", "test", "mock", "streamlit", "openai"]
    return random.choice(words)

def generate_sentence(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Use "gpt-4" or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error generating text: {e}")
        return "Error generating text."



# Function to translate text to Spanish
def translate_to_spanish(text):
    try:
        translated_text = GoogleTranslator(source='en', target='es').translate(text)
        return translated_text
    except Exception as e:
        st.error(f"Error translating text: {e}")
        return "Translation error."

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
    st.subheader(f"Hello, {st.session_state.username}!")
    st.write("""
    To use this app, we need the following permissions:
    
    - **Camera:** To capture your gestures or sign language in real-time.
    - **Microphone:** To capture audio input for interpretation.
    """)

    st.write("""
    Please allow camera and microphone permissions when prompted by your browser.
    - Your camera will be used to capture video input.
    - Your microphone will be used to capture audio input.
    """)

    st.camera_input("Camera Permission Test", key="camera_test_placeholder")
    st.audio_input("Microphone Permission Test", key="mic_test_placeholder")

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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", caption="Live Video Stream", use_container_width=True)

        # Simulated Prediction Logic
        simulated_prediction = mock_word_predictions()  # Simulated prediction
        predicted_words.append(simulated_prediction)

        # If enough words are collected, send them to OpenAI API
        if len(predicted_words) >= 5:  # Example threshold
            predicted_sentence = " ".join(predicted_words)
            predicted_words = []  # Reset words buffer
            
            # Generate sentence using GPT-3.5 Turbo
            llm_output = generate_sentence(predicted_sentence)
            
            # Translate to Spanish
            translated_sentence = translate_to_spanish(llm_output)
            
            # Update session state
            st.session_state.predicted_sentence = llm_output
            st.success(f"English: {llm_output}")
            st.success(f"Español: {translated_sentence}")

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
