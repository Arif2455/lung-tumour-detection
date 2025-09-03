import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

# Load trained model
model = tf.keras.models.load_model("lung_tumour_model.h5")

st.title("🫁 Lung Tumour Detection")

uploaded_file = st.file_uploader("Upload a lung X-ray", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Preprocess image
    img = load_img(uploaded_file, target_size=(224, 224))  # resize to model input size
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    label = "Tumour" if prediction[0][0] > 0.5 else "Normal"

    st.image(uploaded_file, caption=f"Prediction: {label}", use_column_width=True)
