import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, Flatten, MaxPooling2D, Dense
from tensorflow.keras.regularizers import L2
import numpy as np
from PIL import Image
from pathlib import Path
import base64
import os

st.set_page_config(page_title="Fire Detection", page_icon="🔥", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_WEIGHTS_PATH = PROJECT_DIR / "model.weights.h5"
ASSETS_DIR = BASE_DIR / "assets"
LOCAL_BG_PATH = ASSETS_DIR / "background.jpg"
REMOTE_BG_URL = "https://images.unsplash.com/photo-1602980085374-7e743fff3cc6?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
USE_REMOTE_BG = os.getenv("USE_REMOTE_BG", "").lower() in {"1", "true", "yes"}
CSS_PATH = BASE_DIR / "style.css"

if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

def get_background_image_css():
    if USE_REMOTE_BG:
        return f'url("{REMOTE_BG_URL}")'
    if LOCAL_BG_PATH.exists():
        encoded = base64.b64encode(LOCAL_BG_PATH.read_bytes()).decode("ascii")
        return f'url("data:image/jpeg;base64,{encoded}")'
    return f'url("{REMOTE_BG_URL}")'

st.markdown(f"<style>:root {{ --app-bg-image: {get_background_image_css()}; }}</style>", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🔥 Fire Detection</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="sub-title">Upload an image to classify whether it contains Fire or not.</h3>', unsafe_allow_html=True)

@st.cache_resource
def load_classification_model():
    if not MODEL_WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Missing weights file: {MODEL_WEIGHTS_PATH}")
    model = Sequential([
        Input((224, 224, 3)),
        
        Conv2D(activation='relu', filters=32, kernel_size=(3, 3), padding='same'),
        Conv2D(activation='relu', filters=32, kernel_size=(3, 3), padding='same'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Dropout(0.25),
        
        Conv2D(activation='relu', filters=64, kernel_size=(3, 3), padding='same'),
        Conv2D(activation='relu', filters=64, kernel_size=(3, 3), padding='same'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Dropout(0.25),
        
        Conv2D(activation='relu', filters=128, kernel_size=(3, 3), padding='same'),
        Conv2D(activation='relu', filters=128, kernel_size=(3, 3), padding='same'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Dropout(0.3),
        
        Conv2D(activation='relu', filters=256, kernel_size=(3, 3), padding='same'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Dropout(0.3),
        
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer = L2(1e-5)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer = L2(1e-5)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    model.load_weights(MODEL_WEIGHTS_PATH)
    return model

try:
    model = load_classification_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Classes based on the dataset: 0: Smoke, 1: Fire, 2: Non Fire
class_names = ['Smoke', 'Fire', 'Non Fire']

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    with st.spinner('Classifying image...'):
        try:
            # Preprocessing
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img = image.resize((224, 224))
            img_array = np.array(img)
            img_array = img_array.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array)
            prob_array = predictions[0]
            pred_idx = int(np.argmax(prob_array))
            if pred_idx == 1:
                pred_class = 'Fire'
                confidence = prob_array[1] * 100
            elif pred_idx == 0:
                pred_class = 'Smoke'
                confidence = prob_array[0] * 100
            else:
                pred_class = 'Safe'
                confidence = prob_array[2] * 100
            
            st.markdown(f'<h3 class="prediction-text">Prediction: {pred_class}</h3>', unsafe_allow_html=True)
            st.markdown(f'<h4 class="prediction-text">Confidence: {confidence:.2f}%</h4>', unsafe_allow_html=True)
            
            if pred_idx == 1: # fire
                st.markdown('<div class="alert-danger">🚨 DANGER: Fire Detected!</div>', unsafe_allow_html=True)
            elif pred_idx == 0: # smoke
                st.markdown('<div class="alert-warning">🌫️ NO FIRE: Smoke Detected.</div>', unsafe_allow_html=True)
            else: # non fire
                st.markdown('<div class="alert-safe">🛡️ NO FIRE: No Fire Detected.</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
