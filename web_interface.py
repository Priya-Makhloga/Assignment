
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('saved_model/model.h5')

st.title("Image Classifier")
uploaded = st.file_uploader("Upload an image")

if uploaded:
    img = Image.open(uploaded).convert('L').resize((28, 28))
    arr = np.array(img).reshape(1, 28, 28, 1) / 255.0
    pred = np.argmax(model.predict(arr), axis=-1)[0]
    st.image(img)
    st.write(f"Prediction: {pred}")
