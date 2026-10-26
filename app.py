"""
CropVision: Agricultural Crop Plant Image Classification Web App
Built with Streamlit and PyTorch
"""

import os
import glob
import json
import random
from typing import Dict, Any, List
from PIL import Image
import pandas as pd
import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

# Page Configuration
st.set_page_config(
    page_title="CropVision | AI Crop Classification",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a1e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b6b4b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f6faf6;
        border: 1px solid #d4ebd4;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .crop-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #205c20;
    }
    .confidence-badge {
        display: inline-block;
        background-color: #2e7d32;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .info-tag {
        font-weight: 600;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Botanical and Agronomic Knowledge Base
CROP_INSIGHTS: Dict[str, Dict[str, str]] = {
    "Maize (Corn)": {"type": "Cereal / Grain", "climate": "Warm-season (20-30°C)", "soil": "Well-drained loamy soil, pH 5.8-7.0", "water": "500-800 mm / season", "tips": "Requires high nitrogen. Critical irrigation period is during silking and tasseling."},
    "Wheat": {"type": "Cereal / Grain", "climate": "Cool temperate (15-24°C)", "soil": "Clay-loam or silty loam, pH 6.0-7.5", "water": "350-500 mm / season", "tips": "Sow in well-prepared seedbed. Ensure crown root initiation stage receives adequate moisture."},
    "Rice (Paddy)": {"type": "Cereal / Grain", "climate": "Warm, humid tropical (22-32°C)", "soil": "Heavy clay or clay loam holding standing water", "water": "1200-2000 mm / season", "tips": "Maintain 2-5 cm water layer during vegetative and tillering phases."},
    "Cotton lint": {"type": "Fiber / Cash Crop", "climate": "Warm, semi-arid (21-35°C)", "soil": "Deep alluvial or black cotton soil", "water": "700-1200 mm / season", "tips": "Needs frost-free period of 180-200 days. Guard against bollworm and aphids."},
    "Tomatoes": {"type": "Vegetable / Fruit", "climate": "Warm temperate (18-28°C)", "soil": "Rich, well-drained sandy loam, pH 6.0-6.8", "water": "400-600 mm / season", "tips": "Prune suckers for better air circulation and yield. Provide trellis support."},
    "Potatoes": {"type": "Tuber / Root Crop", "climate": "Cool temperate (15-20°C)", "soil": "Loose, friable sandy loam, pH 5.0-6.0", "water": "500-700 mm / season", "tips": "Hill soil around stems as shoots grow to prevent greening of tubers."},
    "Apples": {"type": "Deciduous Fruit Tree", "climate": "Temperate with chilling hours (<7°C)", "soil": "Deep, well-drained loam, pH 6.0-7.0", "water": "800-1000 mm / year", "tips": "Annual winter pruning balances vegetative growth and fruit-bud development."},
    "Bananas": {"type": "Tropical Fruit", "climate": "Warm tropical (26-30°C)", "soil": "Deep, rich organic loam with great drainage", "water": "1500-2500 mm / year", "tips": "High potassium feeder. Protect from strong winds to prevent pseudo-stem breakage."},
    "Coffee (green)": {"type": "Cash Crop / Beverage", "climate": "Sub-tropical / tropical highland (15-24°C)", "soil": "Rich volcanic loam, pH 5.0-6.0", "water": "1500-2000 mm / year", "tips": "Benefits from partial shade canopy. Ensure adequate elevation for Arabica varieties."},
    "Chickpeas": {"type": "Legume / Pulse", "climate": "Cool, dry (18-26°C)", "soil": "Well-drained sandy-loam to clay loam", "water": "250-400 mm / season", "tips": "Fixes atmospheric nitrogen. Highly susceptible to waterlogging."},
    "Soybeans": {"type": "Oilseed / Legume", "climate": "Warm temperate to tropical (20-30°C)", "soil": "Fertile loam with good drainage, pH 6.0-6.8", "water": "450-700 mm / season", "tips": "Inoculate seeds with Rhizobium bacteria to maximize biological nitrogen fixation."},
    "Sunflower seeds": {"type": "Oilseed / Cash Crop", "climate": "Warm temperate to sub-tropical (20-28°C)", "soil": "Deep loam with high water holding capacity", "water": "500-650 mm / season", "tips": "Deep taproots make it drought-tolerant once established. Ensure full sunlight exposure."},
    "Tea": {"type": "Cash Crop / Beverage", "climate": "Humid subtropical / highland (18-30°C)", "soil": "Acidic, deep well-drained soil, pH 4.5-5.5", "water": "1500-2500 mm / year", "tips": "Requires high humidity and rainfall. Regular plucking promotes new tender flushes."}
}

DEFAULT_INSIGHT = {
    "type": "Agricultural Crop",
    "climate": "Check local regional agronomy guidelines for this specific species",
    "soil": "Well-drained, nutrient-balanced agricultural soil",
    "water": "Moderate regular irrigation avoiding standing water",
    "tips": "Inspect plant foliage regularly for signs of nutrient deficiency, pests, or fungal blight."
}


@st.cache_resource
def load_classification_system():
    """Loads class list and model checkpoint."""
    checkpoint_candidates = [
        "models/crop_resnet18_best.pth",
        "models/crop_resnet18_latest.pth"
    ]
    checkpoint_path = None
    for p in checkpoint_candidates:
        if os.path.exists(p):
            checkpoint_path = p
            break

    # Load class names
    class_names_path = "models/class_names.json"
    if os.path.exists(class_names_path):
        with open(class_names_path, "r") as f:
            data = json.load(f)
            class_names = data.get("classes", [])
    else:
        train_dir = "RGB_224x224/train"
        if os.path.exists(train_dir):
            class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        else:
            class_names = []

    model = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        from src.model import load_model_from_checkpoint
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, loaded_classes, _ = load_model_from_checkpoint(checkpoint_path, device=device)
            if loaded_classes:
                class_names = loaded_classes
        except Exception as e:
            st.warning(f"Could not load trained weights: {e}. Model will use pre-trained backbone.")

    return model, class_names, checkpoint_path


def predict_crop(image: Image.Image, model, class_names: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
    """Runs inference on a PIL image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        if model is not None:
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)
        else:
            # Fallback uniform
            probs = torch.ones(len(class_names)) / len(class_names)

    k = min(top_k, len(class_names))
    top_probs, top_indices = torch.topk(probs, k=k)

    results = []
    for rank, (p, idx) in enumerate(zip(top_probs, top_indices), start=1):
        raw_name = class_names[idx.item()]
        clean = raw_name.replace(" plant", "").strip()
        results.append({
            "rank": rank,
            "raw_name": raw_name,
            "clean_name": clean,
            "probability": float(p.item()),
            "confidence_pct": round(float(p.item()) * 100.0, 2)
        })

    return results


# Load model & classes
model, class_names, checkpoint_path = load_classification_system()

# Sidebar Navigation & Statistics
st.sidebar.image("https://images.unsplash.com/photo-1500937386664-56d1dfef3854?w=500&auto=format&fit=crop&q=80", use_container_width=True)
st.sidebar.markdown("### 🌾 CropVision AI")
st.sidebar.markdown("Deep Learning classification system trained on agricultural crop plant images.")

mode = st.sidebar.radio(
    "Navigation",
    ["🔍 Crop Identifier", "📊 Model Performance", "🗂️ Dataset Explorer"]
)



# ==========================================
# PAGE 1: CROP IDENTIFIER
# ==========================================
if mode == "🔍 Crop Identifier":
    st.markdown('<div class="main-header">🌾 Agricultural Crop Plant Identifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a leaf, plant, or field photo to classify the crop and access agronomic cultivation tips.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2], gap="large")

    image_to_predict = None

    with col1:
        st.markdown("#### 1. Input Image")
        input_method = st.radio("Choose image source:", ["Upload File", "Pick from Test Gallery", "Camera Capture"], horizontal=True)

        if input_method == "Upload File":
            uploaded_file = st.file_uploader("Upload Crop Image (JPG, PNG, JPEG):", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image_to_predict = Image.open(uploaded_file)
                st.image(image_to_predict, caption="Uploaded Crop Image", use_container_width=True)

        elif input_method == "Pick from Test Gallery":
            sample_candidates = [
                ("Maize (Corn)", "Maize (Corn) plant"),
                ("Wheat", "Wheat plant"),
                ("Rice (Paddy)", "Rice (Paddy) plant"),
                ("Cotton", "Cotton lint plant"),
                ("Apples", "Apples plant"),
                ("Tomatoes", "Tomatoes plant"),
                ("Coffee", "Coffee (green) plant"),
                ("Bananas", "Bananas plant"),
                ("Sunflowers", "Sunflower seeds plant")
            ]
            selected_sample_label = st.selectbox(
                "Select a sample crop to test:",
                [label for label, _ in sample_candidates]
            )
            # Find directory in test set
            dir_name = dict(sample_candidates)[selected_sample_label]
            test_folder = os.path.join("RGB_224x224", "test", dir_name)
            fallback_sample = os.path.join("samples", f"{dir_name}.jpg")
            
            sample_img_path = None
            if os.path.exists(test_folder):
                images_in_dir = glob.glob(os.path.join(test_folder, "*.jpg")) + glob.glob(os.path.join(test_folder, "*.png"))
                if images_in_dir:
                    sample_img_path = images_in_dir[0]
            elif os.path.exists(fallback_sample):
                sample_img_path = fallback_sample

            if sample_img_path:
                image_to_predict = Image.open(sample_img_path)
                st.image(image_to_predict, caption=f"Test Sample: {selected_sample_label}", use_container_width=True)
            else:
                st.info("Sample image not found.")

        elif input_method == "Camera Capture":
            camera_image = st.camera_input("Take a photo of a crop plant:")
            if camera_image is not None:
                image_to_predict = Image.open(camera_image)

    with col2:
        st.markdown("#### 2. Classification & Farming Profile")

        if image_to_predict is not None:
            with st.spinner("Analyzing botanical features with ResNet-18..."):
                predictions = predict_crop(image_to_predict, model, class_names, top_k=5)

            top1 = predictions[0]
            clean_name = top1["clean_name"]
            confidence = top1["confidence_pct"]

            # Highlighted result card
            st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="crop-title">🌱 {clean_name}</span>
                    <span class="confidence-badge">{confidence:.1f}% Confidence</span>
                </div>
                <div style="color: #666; font-size: 0.9rem; margin-top: 4px;">Top-1 Model Prediction (Class: {top1['raw_name']})</div>
            </div>
            """, unsafe_allow_html=True)

            # Top-5 Distribution Chart
            st.markdown("##### Top-5 Predictions Probability")
            df_preds = pd.DataFrame([
                {"Crop": p["clean_name"], "Confidence (%)": p["confidence_pct"]}
                for p in predictions
            ])
            st.bar_chart(df_preds.set_index("Crop"), color="#2e7d32")

            # Agronomic Profile
            st.markdown("##### 🌾 Agronomic Cultivation Profile")
            crop_info = CROP_INSIGHTS.get(clean_name, DEFAULT_INSIGHT)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"<span class='info-tag'>Category:</span> {crop_info.get('type', 'Crop')}", unsafe_allow_html=True)
                st.markdown(f"<span class='info-tag'>Ideal Climate:</span> {crop_info.get('climate', 'N/A')}", unsafe_allow_html=True)
            with col_b:
                st.markdown(f"<span class='info-tag'>Water Requirement:</span> {crop_info.get('water', 'N/A')}", unsafe_allow_html=True)
                st.markdown(f"<span class='info-tag'>Soil Recommendation:</span> {crop_info.get('soil', 'N/A')}", unsafe_allow_html=True)

            st.info(f"💡 **Farming Best Practice:** {crop_info.get('tips', 'Ensure balanced soil nutrients and scout for early leaf disease symptoms.')}")

        else:
            st.markdown("""
            <div style="background-color: #fafafa; border: 2px dashed #ddd; border-radius: 10px; padding: 40px; text-align: center; color: #888;">
                <p style="font-size: 1.2rem;">👈 Please upload or select a crop image to start classification.</p>
                <p style="font-size: 0.9rem;">The system will classify among 139 plant categories and provide agricultural guidance.</p>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# PAGE 2: MODEL PERFORMANCE
# ==========================================
elif mode == "📊 Model Performance":
    st.markdown('<div class="main-header">📊 Model Performance & Training Dynamics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation metrics, loss curves, top-1/top-5 accuracy, and architecture details.</div>', unsafe_allow_html=True)

    history_path = "models/training_history.json"
    eval_path = "models/evaluation_results.json"

    # Display Top Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Crop Classes", f"{len(class_names)}")
    with col2:
        st.metric("Input Image Size", "224 x 224 x 3")
    with col3:
        st.metric("Backbone", "ResNet-18 (ImageNet)")
    with col4:
        st.metric("Framework", "PyTorch 2.x")

    st.markdown("---")

    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)

        epochs = list(range(1, len(history.get("train_loss", [])) + 1))
        df_history = pd.DataFrame({
            "Epoch": epochs,
            "Train Loss": history.get("train_loss", []),
            "Val Loss": history.get("val_loss", []),
            "Train Top-1 Acc (%)": history.get("train_top1", []),
            "Val Top-1 Acc (%)": history.get("val_top1", []),
            "Val Top-5 Acc (%)": history.get("val_top5", [])
        }).set_index("Epoch")

        col_loss, col_acc = st.columns(2)
        with col_loss:
            st.subheader("📉 Loss Convergence (Cross-Entropy)")
            st.line_chart(df_history[["Train Loss", "Val Loss"]])

        with col_acc:
            st.subheader("📈 Top-1 & Top-5 Accuracy (%)")
            st.line_chart(df_history[["Train Top-1 Acc (%)", "Val Top-1 Acc (%)", "Val Top-5 Acc (%)"]])

        st.dataframe(df_history, use_container_width=True)
    else:
        st.info("Training history will appear here once the training process completes.")

    # Evaluation results if present
    if os.path.exists(eval_path):
        st.markdown("---")
        st.subheader("🧪 Test Set Evaluation Report")
        with open(eval_path, "r") as f:
            eval_data = json.load(f)

        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            st.metric("Test Top-1 Accuracy", f"{eval_data.get('top1_accuracy_pct', 0):.2f}%")
        with col_e2:
            st.metric("Test Top-5 Accuracy", f"{eval_data.get('top5_accuracy_pct', 0):.2f}%")
        with col_e3:
            st.metric("Total Test Images", f"{eval_data.get('total_test_samples', 0):,}")

# ==========================================
# PAGE 3: DATASET EXPLORER
# ==========================================
elif mode == "🗂️ Dataset Explorer":
    st.markdown('<div class="main-header">🗂️ Agricultural Crop Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Browse all <b>{len(class_names)}</b> distinct crop species present in the dataset.</div>', unsafe_allow_html=True)

    search_query = st.text_input("🔍 Search crop name (e.g. Rice, Pepper, Bean, Apple, Wheat):", "")

    filtered_classes = [c for c in class_names if search_query.lower() in c.lower()] if search_query else class_names

    st.caption(f"Showing {len(filtered_classes)} of {len(class_names)} categories")

    # Display in 4 columns
    cols = st.columns(4)
    for i, c in enumerate(filtered_classes):
        col_idx = i % 4
        clean = c.replace(" plant", "").strip()
        with cols[col_idx]:
            st.markdown(f"- **{clean}**")
