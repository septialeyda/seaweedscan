import os
import tempfile
import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import gdown

MODEL_URL = "https://drive.google.com/uc?id=1XAUiORzmfbHyrogIi3viw_jt90ASboSE"
MODEL_PATH = "seaweed_yolo8.pt"

# Load model only once
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading YOLOv8 model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return YOLO(MODEL_PATH)

model = load_model()

# Page setup
st.set_page_config(page_title="SeaweedScan 🌿", layout="wide")

# YOLOv8 model inference
def model_prediction(uploaded_file):
    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    results = model.predict(source=temp_path, conf=0.25)
    os.remove(temp_path)

    if results:
        if hasattr(results[0], "probs") and results[0].probs is not None:
            result_index = int(np.argmax(results[0].probs.data))
            class_name = ["Acanthophora", "Caulerpa", "Eucheuma", "Gracilaria", "Halimeda",
                          "Padina", "Sargassum", "Turbinaria", "Ulva"]
            predicted_label = class_name[result_index]
            confidence = float(results[0].probs.data[result_index])
        else:
            predicted_label = "Unknown"
            confidence = 0.0

        annotated_image = results[0].plot()
        return annotated_image, predicted_label, confidence

    return None, None, None

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Seaweed Recognition"])

# Home Page
if app_mode == "🏠 Home":
    st.markdown("<h1 style='text-align: center;'>🌊 SeaweedScan</h1>", unsafe_allow_html=True)
    
    try:
        st.image("home.jpg", use_container_width=True)
    except:
        st.warning("📷 `home.jpg` not found. Add it to your repo to display a banner image.")
    
    st.markdown("""
        <div style='padding: 1.5em; border-radius: 10px; background-color: rgba(255,255,255,0.05);'>
            <h3 style="color:#70e000;">Welcome to SeaweedScan 🌿</h3>
            <p style="font-size: 1.1em;">Discover the fascinating world of seaweed with our AI-powered image recognition system.</p>

            <hr style="border-color: #444;">
            <h4>🔧 How It Works</h4>
            <ul>
                <li>📸 Snap a photo of the seaweed specimen</li>
                <li>📤 Upload to SeaweedScan</li>
                <li>🤖 Let our AI recognize it</li>
                <li>🌍 Contribute to marine research</li>
            </ul>

            <h4>💡 Why Use SeaweedScan?</h4>
            <ul>
                <li>✅ Accurate recognition</li>
                <li>🧪 Citizen science support</li>
                <li>🌐 Community-driven project</li>
                <li>😌 Easy to use interface</li>
            </ul>
            
            <p>👉 Use the <strong>Seaweed Recognition</strong> tab to begin!</p>
        </div>
    """, unsafe_allow_html=True)

# Recognition Page
elif app_mode == "🔍 Seaweed Recognition":
    st.markdown("### 📤 Upload your seaweed image")
    uploaded_file = st.file_uploader("Supported formats: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='📷 Uploaded Image', use_container_width=True)
        if st.button("🚀 Predict"):
            with st.spinner("Analyzing image..."):
                image, label, confidence = model_prediction(uploaded_file)
                if label is not None:
                    st.image(image, caption="🔍 Detected Seaweed", use_container_width=True)
                    st.success(f"✅ It's a **{label}** with confidence **{confidence:.2f}**")
                else:
                    st.warning("⚠️ No seaweed detected or image not suitable.")
