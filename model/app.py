import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the model
model_path = "resnet_best_model.h5"  # You can change to custom_cnn.h5 if needed
model = tf.keras.models.load_model(model_path)

# Class names
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Streamlit UI
st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="centered")
st.title("Brain Tumor MRI Image Classifier")
st.write("Upload an MRI image and the model will predict the type of brain tumor.")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded MRI', use_column_width=True)

    # Preprocess the image
    img_size = (224, 224)
    img_array = np.array(image.resize(img_size)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"🧾 **Predicted Tumor Type:** {predicted_class.capitalize()}")
