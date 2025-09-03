import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

# 🎨 Streamlit page config
st.set_page_config(
    page_title="Lung Tumour Detection 🫁",
    page_icon="🩻",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Load model
model = tf.keras.models.load_model("lung_tumour_model.h5")

# 🌟 Title and description
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>🫁 Lung Tumour Detection</h1>
    <p style='text-align: center; color: gray;'>
    Upload a chest X-ray and let the AI model analyze it for potential tumour presence.<br>
    <i>⚠️ For educational purposes only — not a medical diagnosis.</i>
    </p>
    """,
    unsafe_allow_html=True,
)

# Sidebar info
st.sidebar.header("📂 About this App")
st.sidebar.info(
    """
    - Built with **Streamlit** & **TensorFlow**
    - Upload X-ray in `.png`, `.jpg`, `.jpeg`
    - Model will classify as **Normal** or **Tumour**
    """
)

# File uploader
uploaded_file = st.file_uploader("📤 Upload a Lung X-ray", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Preprocess image
    img = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    st.write("🔎 **Raw prediction values:**", prediction)

    if prediction.shape[-1] == 1:
        label = "🟥 Tumour Detected" if prediction[0][0] > 0.5 else "🟩 Normal"
        confidence = float(prediction[0][0]) if label == "🟥 Tumour Detected" else 1 - float(prediction[0][0])
    else:
        classes = ["🟩 Normal", "🟥 Tumour Detected"]
        label = classes[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]))

    # Display uploaded image + prediction
    st.image(uploaded_file, caption="Uploaded X-ray", use_column_width=True)

    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 20px;">
            <h2 style="color: {'red' if 'Tumour' in label else 'green'};">{label}</h2>
            <p><b>Confidence:</b> {confidence:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Add progress bar for fun 🎉
    st.progress(confidence)

else:
    st.info("⬆️ Please upload an X-ray to start the analysis.")
