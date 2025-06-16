import os
import streamlit as st
from ultralytics import YOLO
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
import torch
import types

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

st.image("https://raw.githubusercontent.com/Rajesh2015/Food-101-Detection-Capstone-Project/refs/heads/main/application/src/assets/logo.png", width=300)  # Adjust width as needed
st.markdown("### Food Classifier And Bounding Box Generator Trained with Yolo V8\nby **Team CV5**")
st.sidebar.title("📌 About")
st.sidebar.markdown("""
**Food Detection App CV Team 5**  
Using YOLOv8 to detect food items from the **Food-101** dataset.

- Built with [Ultralytics YOLO](https://docs.ultralytics.com)
- Trained on custom dataset

📧 Contact: `dashrajesh49@gmail.com`
""")

# Main area
st.title("🍔 Food Item Detection")
st.markdown("Upload a food image, and click **Predict** to see bounding boxes.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
predict_button = st.button("🔍 Predict")

# Prediction logic
if uploaded_file and predict_button:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Run inference
    results = model(image_np)
    annotated_img = results[0].plot()

    # Display
    st.image(annotated_img, caption="🔎 Predicted Image", use_column_width=True)

    # Boxes
    st.subheader("📦 Detected Objects")
    for box in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, score, cls_id = box
        cls_name = model.names[int(cls_id)]
        st.markdown(f"- **{cls_name}** ({score:.2f}) — Box: `[x1: {int(x1)}, y1: {int(y1)}, x2: {int(x2)}, y2: {int(y2)}]`")
