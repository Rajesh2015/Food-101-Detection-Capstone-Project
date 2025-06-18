import os
import streamlit as st
from ultralytics import YOLO
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
import torch
import types
from PIL import Image
import numpy as np

# --- Patch torch_safe_load to force weights_only=False ---
from ultralytics.nn import tasks

original_safe_load = tasks.torch_safe_load

def patched_safe_load(file, device=None):
    ckpt=torch.load(file, map_location=torch.device("cpu"), weights_only=False)
    return ckpt, file  # 👈 match expected return signature

tasks.torch_safe_load = patched_safe_load

# Allowlist DetectionModel so torch.load works with pickled class
add_safe_globals([DetectionModel])


# Load model once
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "yolo_model_aug.pt")
    return YOLO(model_path)

model = load_model()
model.to("cpu")


# Sidebar

# Centered logo at the top
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://raw.githubusercontent.com/Rajesh2015/Food-101-Detection-Capstone-Project/refs/heads/main/application/src/assets/logo.png" width="200">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("### Food Classifier And Bounding Box Generator Trained with Yolo V8\nby **Team CV5**")
st.sidebar.title("📌 About")
st.sidebar.markdown("""
**Food Detection App Team CV5**  
Using YOLOv8 to detect food items from the **Food-101** dataset.

- Built with [Ultralytics YOLO](https://docs.ultralytics.com)
- Trained on custom dataset

📧 Contact: `dashrajesh49@gmail.com`
""")

# Main area
st.markdown("## 🍽️ Food Item Detection")
st.markdown("Upload an image of a food item and click **🔍 Predict** to detect and classify items.")
st.markdown("---")

# File uploader
col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
with col2:
    # Add vertical space to center the button with file uploader height
    st.write("")  # Adds spacing (you can duplicate it or use <br> for more control)
    st.write("")  
    predict_button = st.button("🔍 Predict", use_container_width=True)


# Prediction logic
if uploaded_file and predict_button:
    with st.spinner("🔍 Running prediction... Please wait."):
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)

            # Run inference
            results = model(image_np)
    annotated_img = results[0].plot()

    # Display
    st.image(annotated_img, caption="🧠 Model Output with Detected Boxes", use_column_width=True)
    st.success(f"✅ {len(results[0].boxes)} object(s) detected.")
    st.subheader("📦 Detected Items")
    for box in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, score, cls_id = box
        cls_name = model.names[int(cls_id)]
        st.markdown(f"""
        - **{cls_name}**  
        🔹 Confidence: `{score:.2f}`  
        🔹 Coordinates: `[x1: {int(x1)}, y1: {int(y1)}, x2: {int(x2)}, y2: {int(y2)}]`
        """)



    # Boxes
    st.subheader("📦 Detected Objects")
    for box in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, score, cls_id = box
        cls_name = model.names[int(cls_id)]
        st.markdown(f"- **{cls_name}** ({score:.2f}) — Box: `[x1: {int(x1)}, y1: {int(y1)}, x2: {int(x2)}, y2: {int(y2)}]`")
