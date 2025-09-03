🫁 Lung Tumour Detection using Medical Image Processing

This project demonstrates the use of medical image processing and deep learning for the detection of lung tumours from chest X-ray images.
A Convolutional Neural Network (CNN) built with TensorFlow/Keras powers the backend, while an interactive Streamlit interface enables real-time predictions.

⚠️ Disclaimer: This tool is for educational and research purposes only. It is not intended for clinical or diagnostic use.

🔬 Overview

Input: Chest X-ray (.png, .jpg, .jpeg)

Processing: Image resizing, normalization, and CNN-based feature extraction

Output: Prediction as either 🟩 Normal or 🟥 Tumour Detected

Interface: Streamlit web application for easy interaction

✨ Key Features

Web-based interface for quick image upload and analysis

Displays confidence scores along with predictions

Shows the uploaded image with predicted label

Implements a standard medical image preprocessing pipeline

🩻 Methodology

Input chest X-ray uploaded by user

Image preprocessing (resizing to 224×224, normalization)

Forward pass through the CNN model (convolution, pooling, dense layers)

Classification into Normal or Tumour with probability score

🛠️ Tech Stack

Language: Python (3.9+)

Deep Learning: TensorFlow, Keras

Interface: Streamlit

Supporting Libraries: NumPy, Pillow, Matplotlib

📂 Project Structure

app.py → Streamlit frontend

lung_tumour_model.h5 → Trained CNN model (HDF5 format)

lung_tumour_model_v2.keras → Trained CNN model (Keras format, recommended)

requirements.txt → Dependencies list

README.md → Documentation

⚙️ Setup & Installation

Clone the repository:
git clone https://github.com/Arif2455/lung-tumour-detection.git
cd lung-tumour-detection

Create and activate virtual environment:
conda create -n lung python=3.9 -y
conda activate lung

Install dependencies:
pip install -r requirements.txt

Run the application:
streamlit run app.py

📊 Model Details

Architecture: Custom CNN

Input Shape: 224 × 224 × 3

Task: Binary classification (Normal vs Tumour)

Exported Formats: .h5 (legacy) and .keras (recommended)

🚀 Future Improvements

Add Grad-CAM heatmaps for visual explainability

Expand to CT and MRI modalities

Deploy on Streamlit Cloud or Hugging Face Spaces

Improve generalization with larger datasets

🙌 Acknowledgements

Publicly available chest X-ray datasets (Kaggle, NIH, etc.)

TensorFlow, Keras, and Streamlit open-source communities

Research in AI-driven medical imaging

📜 License

Licensed under the MIT License. See the LICENSE file for details.
