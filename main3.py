import os
import tempfile
import numpy as np
from PIL import Image
import gdown
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="SeaweedScan 🌿", layout="wide")

MODEL_URL = "https://drive.google.com/uc?id=1OQtARxZnQDS5UzS8vukSQBiQktIDt0Ka"
MODEL_PATH = "seaweed-9.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading YOLOv8 model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return YOLO(MODEL_PATH)

model = load_model()

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

st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Seaweed Recognition"])

if app_mode == "🏠 Home":
    st.markdown("## 🌊 SeaweedScan")

    try:
        st.image("home.jpg", use_container_width=True)
    except:
        st.warning("📷 'home.jpg' not found. Add it to your repo to display a banner image.")

    st.markdown("### 🥬 Welcome to SeaweedScan")
    st.markdown(
        "Discover the fascinating world of seaweed with our AI-powered image recognition system. "
        "Our mission is to make marine biodiversity accessible and interactive."
    )

    st.markdown("---")
    st.markdown("### 🔧 How It Works")
    st.markdown("""
- 📸 Snap a photo of the seaweed specimen  
- 📤 Upload it to SeaweedScan  
- 🤖 Let our AI recognize it  
- 🌍 Contribute to marine research  
    """)

    st.markdown("---")
    st.markdown("### 💡 Why Use SeaweedScan?")
    st.markdown("""
- ✅ Accurate recognition  
- 🧪 Supports citizen science  
- 🌐 Community-driven  
- 😌 Easy to use  
    """)

    st.info("👉 Switch to the **Seaweed Recognition** tab to begin!")
    st.caption("App version: v1.0.0 | Developed by BD Indonesia Team")

elif app_mode == "🔍 Seaweed Recognition":
    st.markdown("## 🔍 Seaweed Recognition")
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

    st.markdown("---")
    st.caption("App version: v1.0.0 | Developed by BD Indonesia Team")
