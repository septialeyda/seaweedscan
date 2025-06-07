import os
import tempfile
import streamlit as st
from ultralytics import YOLO
import numpy as np

# Page setup
st.set_page_config(page_title="SeaweedScan 🌿", layout="wide")

# YOLOv8 model inference
def model_prediction(uploaded_file):
    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    model = YOLO('seaweed_yolo8.pt')
    results = model.predict(source=temp_path, conf=0.25)
    os.remove(temp_path)

    if results and hasattr(results[0], "probs") and results[0].probs is not None:
        result_index = int(np.argmax(results[0].probs.data))
        class_name = ["Acanthophora", "Caulerpa", "Eucheuma", "Gracilaria", "Halimeda", "Padina", "Sargassum", "Turbinaria", "Ulva"]
        predicted_label = class_name[result_index]
        confidence = float(results[0].probs.data[result_index])
        annotated_image = results[0].plot()
        return annotated_image, predicted_label, confidence
    else:
        return None, None, None

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Seaweed Recognition"])

# Home Page
if app_mode == "🏠 Home":
    st.markdown("<h1 style='text-align: center;'>🌊 SeaweedScan</h1>", unsafe_allow_html=True)
    st.image("D:/Blue-Dragon/Deploy/home.jpg", use_container_width=True)
    st.markdown("""
        <div style='padding:20px; background-color:#e8f5e9; border-radius:10px'>
        <h3>Welcome to SeaweedScan, your reliable seaweed image recognition system! 🥬</h3>
        <p>Discover the fascinating world of seaweed with our AI-powered image recognition system.</p>

        ### 🔧 How It Works
        - 📸 Snap a photo of the seaweed specimen.
        - 📤 Upload to SeaweedScan.
        - 🤖 Let our AI recognize it.
        - 🌍 Contribute to marine research!

        ### 💡 Why Use SeaweedScan?
        - ✅ Accurate recognition
        - 🧪 Citizen science support
        - 🌐 Community-driven project
        - 😌 Easy to use interface

        <br>
        👉 Go to the <strong>Seaweed Recognition</strong> page to begin!
        </div>
    """, unsafe_allow_html=True)

# Recognition Page
elif app_mode == "🔍 Seaweed Recognition":
    st.markdown("<h1>🔍 Seaweed Recognition</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📁 Upload your seaweed image", type=["jpg", "jpeg", "png"])

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
