# 🫁 LUNG TUMOUR DETECTION USING MEDICAL IMAGE PROCESSING

This project demonstrates the use of **medical image processing** and **deep learning** for the detection of lung tumours from chest X-ray images.  
A **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras** powers the backend, while an interactive **Streamlit interface** enables real-time predictions.

⚠️ **Disclaimer**: This tool is for **educational and research purposes only**. It is **not intended for clinical or diagnostic use**.

## 🔬 OVERVIEW
- **Input**: Chest X-ray (`.png`, `.jpg`, `.jpeg`)  
- **Processing**: Image resizing, normalization, and CNN-based feature extraction  
- **Output**: Prediction as either 🟩 Normal or 🟥 Tumour Detected  
- **Interface**: Streamlit web application for easy interaction  

## ✨ KEY FEATURES
- Web-based interface for quick image upload and analysis  
- Displays **confidence scores** along with predictions  
- Shows the **uploaded image with predicted label**  
- Implements a standard **medical image preprocessing pipeline**  

## 🩻 METHODOLOGY
1. Input chest X-ray uploaded by user  
2. Image preprocessing (resizing to 224×224, normalization)  
3. Forward pass through the **CNN model** (convolution, pooling, dense layers)  
4. Classification into **Normal** or **Tumour** with probability score  

## 🛠️ TECH STACK
- **Language**: Python (3.9+)  
- **Deep Learning**: TensorFlow, Keras  
- **Interface**: Streamlit  
- **Supporting Libraries**: NumPy, Pillow, Matplotlib  

## 📂 PROJECT STRUCTURE
- **app.py** → Streamlit frontend  
- **lung_tumour_model.h5** → Trained CNN model (HDF5 format)  
- **lung_tumour_model_v2.keras** → Trained CNN model (Keras format, recommended)  
- **requirements.txt** → Dependencies list  
- **README.md** → Documentation  

## ⚙️ SETUP & INSTALLATION
1. Clone the repository:  
   ```bash
   git clone https://github.com/YourUsername/lung-tumour-detection.git
   cd lung-tumour-detection
