"""
Plant Disease Detection Using AI/ML
BCA Third-Year Summer Project - University Web Application

Main Streamlit Application
- Agriculture + Technology Design System
- Streamlit cached model loading
- Real-time Deep Learning image classification
- Top-3 prediction confidence ranking
- Comprehensive agronomic disease knowledge base
- Live analytical dashboard & university project documentation
"""

import os
import io
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src.preprocessing import load_and_preprocess_image, ImageValidationError
from src.prediction import (
    is_model_available,
    load_trained_model,
    load_class_names,
    predict_leaf_disease,
    DEFAULT_MODEL_PATH,
    DEFAULT_CLASSES_PATH
)
from src.disease_info import (
    get_disease_info,
    get_all_supported_crops,
    get_diseases_for_crop,
    get_database_statistics,
    parse_raw_class_name,
    DISCLAIMER_TEXT
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Detection Using AI/ML",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Agriculture & Tech Aesthetics
st.markdown("""
<style>
    /* Primary Colors: Emerald, Forest, Soft Mint & Slate */
    :root {
        --primary-green: #2d6a4f;
        --secondary-green: #52b788;
        --light-bg: #f8faf9;
        --card-bg: #ffffff;
        --border-color: #e2e8f0;
        --text-dark: #1e293b;
        --text-muted: #64748b;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }

    /* Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 60%, #40916c 100%);
        color: #ffffff !important;
        padding: 2rem 2.2rem;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(27, 67, 50, 0.15);
        margin-bottom: 1.8rem;
    }
    .hero-banner h1 {
        color: #ffffff !important;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        color: #d8f3dc !important;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    /* Status Pills */
    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        font-weight: 600;
        font-size: 0.85rem;
        border-radius: 50px;
        letter-spacing: 0.3px;
    }
    .badge-healthy {
        background-color: #d8f3dc;
        color: #1b4332;
        border: 1px solid #74c69d;
    }
    .badge-diseased {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #f87171;
    }

    /* Result Metric Cards */
    .result-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 1.2rem;
    }
    .result-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.3rem;
    }
    .result-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1b4332;
    }

    /* Info Callouts */
    .disclaimer-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-top: 1.2rem;
        font-size: 0.88rem;
        color: #78350f;
    }

    /* Sidebar Clean Styling */
    .sidebar-project-tag {
        background-color: #d8f3dc;
        color: #1b4332;
        padding: 0.5rem 0.8rem;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# CACHED MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI model weights...")
def get_cached_model(model_path: str = DEFAULT_MODEL_PATH):
    """Loads and caches the neural network model to ensure high responsiveness."""
    return load_trained_model(model_path)


@st.cache_data(show_spinner="Loading dataset classes...")
def get_cached_classes(classes_path: str = DEFAULT_CLASSES_PATH):
    """Loads and caches the class name mapping."""
    return load_class_names(classes_path)


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & SYSTEM INFO
# -----------------------------------------------------------------------------
def render_sidebar():
    logo_path = os.path.join("assets", "logo.png")
    if os.path.isfile(logo_path):
        st.sidebar.image(logo_path, width=80)

    st.sidebar.markdown("### 🌿 Plant Disease AI")
    st.sidebar.markdown(
        '<div class="sidebar-project-tag">🎓 BCA 3rd-Year Project</div>',
        unsafe_allow_html=True
    )

    nav_choice = st.sidebar.radio(
        "Navigation",
        ["🌿 Home", "🔍 Disease Detection", "📊 Dashboard", "📖 About Project"],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### System Information")

    model_ready = is_model_available()
    if model_ready:
        st.sidebar.success("✅ Model: Ready (MobileNetV2)")
    else:
        st.sidebar.warning("⚠️ Model: Not Found")

    st.sidebar.caption("• Architecture: MobileNetV2 (CNN)")
    st.sidebar.caption("• Framework: TensorFlow / Keras")
    st.sidebar.caption("• UI: Streamlit Engine")
    st.sidebar.caption("• Privacy: 100% Local Inference")

    st.sidebar.markdown("---")
    st.sidebar.caption("Developed for BCA Final Year Summer Project Demonstration.")

    return nav_choice


# -----------------------------------------------------------------------------
# PAGE 1: HOME
# -----------------------------------------------------------------------------
def render_home_page():
    st.markdown("""
    <div class="hero-banner">
        <h1>Plant Disease Detection Using AI/ML</h1>
        <p>AI-powered image-based plant health classification for precision agriculture</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 🌾 Welcome to the AI Plant Clinic")
        st.markdown("""
        Crop diseases pose a critical threat to global food security and farmer livelihoods. 
        Early visual diagnosis allows timely, targeted intervention before pathogens spread 
        across entire crop fields.

        This system leverages **Convolutional Neural Networks (CNN)** and **Transfer Learning 
        (MobileNetV2)** to analyze leaf photographs, identify bacterial, fungal, and viral infections, 
        and provide actionable, non-destructive management guidance.
        """)

        st.markdown("#### 🚀 How the System Works")
        step_cols = st.columns(3)
        with step_cols[0]:
            st.info("**1. Upload Leaf**\n\nTake or upload a high-clarity photo of the symptomatic leaf.")
        with step_cols[1]:
            st.info("**2. AI Classification**\n\nMobileNetV2 extracts deep visual lesion features.")
        with step_cols[2]:
            st.info("**3. Diagnosis & Care**\n\nInstant disease diagnosis, confidence, and management.")

        st.markdown("---")
        st.markdown("#### 🛡️ Local & Private Processing")
        st.caption(
            "🔒 **Privacy Guarantee:** All neural network computations execute strictly on your local machine. "
            "No photographs or agricultural data are transmitted to external servers."
        )

    with col2:
        st.markdown("### 🍃 Supported Crops")
        crops = get_all_supported_crops()
        for crop in crops:
            diseases = get_diseases_for_crop(crop)
            with st.expander(f"🌱 {crop} ({len(diseases)} conditions)"):
                for d in diseases:
                    icon = "✅" if d["status"] == "Healthy" else "🔴"
                    st.write(f"{icon} **{d['disease']}** ({d['status']})")

        st.markdown("""
        <div style="background-color:#e8f5e9; padding:1.2rem; border-radius:10px; border:1px solid #c8e6c9;">
            <h4 style="color:#2e7d32; margin-top:0;">⚡ Ready to test?</h4>
            <p style="font-size:0.9rem; color:#1b5e20;">
                Navigate to the <b>Disease Detection</b> tab in the sidebar to upload a leaf photograph.
            </p>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 2: DISEASE DETECTION
# -----------------------------------------------------------------------------
def render_detection_page():
    st.markdown("## 🔍 Plant Disease Detection")
    st.caption("Upload a leaf image or take a snapshot with your device camera for real-time AI diagnosis.")

    # 1. Model Availability Check
    model_exists = is_model_available()
    if not model_exists:
        st.error("⚠️ **Trained Model Not Found**")
        st.markdown("""
        The application requires the trained weights file (`models/plant_disease_model.keras`) 
        and class mapping (`models/class_names.json`) to perform real-time predictions.

        **How to Train the Model:**
        1. Ensure your dataset is located in `dataset/train/` and `dataset/validation/`.
        2. Run the training pipeline in your terminal:
           ```bash
           python train_model.py --epochs 15
           ```
        3. Once training completes, refresh this page to begin real inference!
        """)
        return

    # Load cached model & classes
    try:
        model = get_cached_model()
        class_names = get_cached_classes()
    except Exception as e:
        st.error(f"Error initializing AI model: {e}")
        return

    # 2. Input Method Selector
    input_method = st.radio(
        "Select Image Input Source:",
        ["📁 Upload Image File", "📷 Capture from Camera", "🧪 Select Sample Image"],
        horizontal=True
    )

    image_to_process = None

    if input_method == "📁 Upload Image File":
        uploaded_file = st.file_uploader(
            "Choose a plant leaf image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear, focused photograph of an isolated plant leaf."
        )
        if uploaded_file is not None:
            image_to_process = uploaded_file

    elif input_method == "📷 Capture from Camera":
        camera_file = st.camera_input("Take a photo of a plant leaf")
        if camera_file is not None:
            image_to_process = camera_file

    elif input_method == "🧪 Select Sample Image":
        samples_dir = "samples"
        sample_files = []
        if os.path.isdir(samples_dir):
            sample_files = [
                f for f in os.listdir(samples_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

        if sample_files:
            selected_sample = st.selectbox("Choose a demo sample leaf:", sample_files)
            if selected_sample:
                image_to_process = os.path.join(samples_dir, selected_sample)
        else:
            st.info("No sample images found in 'samples/' folder. You can upload an image above.")

    # 3. Processing and Prediction Interface
    if image_to_process is None:
        st.info("👆 Please upload or capture a plant leaf image to continue.")
        return

    col_img, col_pred = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("#### 📷 Leaf Photograph")
        try:
            # Validate and preprocess image
            preprocessed_tensor, display_pil = load_and_preprocess_image(image_to_process)
            st.image(display_pil, use_container_width=True, caption=f"Uploaded Image ({display_pil.size[0]}x{display_pil.size[1]} px)")

            # Clean Image Details
            st.caption(f"Color Mode: {display_pil.mode} | Normalized Model Input: 224x224 RGB")
        except ImageValidationError as ive:
            st.error(f"❌ {str(ive)}")
            return
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
            return

    with col_pred:
        st.markdown("#### 🔬 AI Inference Analysis")

        detect_btn = st.button("🌿 Detect Disease", type="primary", use_container_width=True)

        if detect_btn:
            with st.spinner("Analyzing leaf pathology with MobileNetV2..."):
                try:
                    result = predict_leaf_disease(
                        model=model,
                        class_names=class_names,
                        preprocessed_tensor=preprocessed_tensor,
                        top_k=3
                    )
                except Exception as e:
                    st.error(f"Inference error: {str(e)}")
                    return

            # Store result in session state for persistence
            st.session_state["last_prediction"] = result
            st.session_state["last_image"] = display_pil

        if "last_prediction" in st.session_state:
            result = st.session_state["last_prediction"]

            crop = result["crop"]
            disease = result["disease"]
            status = result["status"]
            conf = result["confidence"]
            top_preds = result["top_predictions"]
            info = result["disease_info"]

            badge_class = "badge-healthy" if status == "Healthy" else "badge-diseased"
            status_icon = "✅" if status == "Healthy" else "⚠️"

            # Prediction Card
            st.markdown(f"""
            <div class="result-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="result-title">Prediction Result</span>
                    <span class="status-badge {badge_class}">{status_icon} {status}</span>
                </div>
                <div style="margin-top:0.6rem;">
                    <div style="font-size:0.95rem; color:#64748b;">Crop</div>
                    <div class="result-value">{crop}</div>
                </div>
                <div style="margin-top:0.6rem;">
                    <div style="font-size:0.95rem; color:#64748b;">Diagnosis</div>
                    <div style="font-size:1.3rem; font-weight:700; color:#2d6a4f;">{disease}</div>
                </div>
                <div style="margin-top:0.8rem;">
                    <div style="display:flex; justify-content:space-between; font-size:0.9rem; font-weight:600;">
                        <span>Confidence Score</span>
                        <span>{conf:.2f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence visual bar
            st.progress(conf / 100.0)

            # Top 3 Candidates
            st.markdown("##### 📊 Top 3 Candidate Predictions")
            top_df = pd.DataFrame([
                {
                    "Rank": p["rank"],
                    "Condition": f"{p['crop']} - {p['disease']}",
                    "Status": p["status"],
                    "Probability (%)": f"{p['probability']:.2f}%"
                }
                for p in top_preds
            ])
            st.table(top_df)

    # 4. Disease Agricultural Information Details
    if "last_prediction" in st.session_state:
        st.markdown("---")
        result = st.session_state["last_prediction"]
        info = result["disease_info"]

        st.markdown(f"### 📋 Agronomic Advisory: {info['crop']} - {info['disease']}")

        tab_desc, tab_sym, tab_prev, tab_mgmt = st.tabs([
            "📖 Overview",
            "🔍 Visible Symptoms",
            "🛡️ Prevention",
            "💊 Management"
        ])

        with tab_desc:
            st.markdown(f"**Pathogen / Agent:** `{info.get('pathogen_type', 'N/A')}`")
            st.write(info["description"])

        with tab_sym:
            st.markdown("**Key Visible Symptoms:**")
            for s in info["symptoms"]:
                st.markdown(f"- {s}")

        with tab_prev:
            st.markdown("**Preventive Cultural Practices:**")
            for p in info["prevention"]:
                st.markdown(f"- {p}")

        with tab_mgmt:
            st.markdown("**Field Management Recommendations:**")
            for m in info["management"]:
                st.markdown(f"- {m}")

        # Advisory Disclaimer
        st.markdown(f"""
        <div class="disclaimer-box">
            <b>⚠️ Agricultural Advisory Disclaimer:</b><br>
            {info.get('expert_disclaimer', DISCLAIMER_TEXT)}
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 3: DASHBOARD
# -----------------------------------------------------------------------------
def render_dashboard_page():
    st.markdown("## 📊 System Analytics & Evaluation Dashboard")
    st.caption("Real performance metrics, dataset distribution, and neural network evaluation charts.")

    # High-level KPIs
    db_stats = get_database_statistics()
    model_ready = is_model_available()

    # Load test metrics if available
    metrics_path = os.path.join("results", "metrics.json")
    test_metrics = None
    if os.path.isfile(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                test_metrics = json.load(f)
        except Exception:
            test_metrics = None

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric("Total Crops Supported", db_stats["total_crops"])
    with kpi2:
        st.metric("Total Disease Classes", db_stats["total_classes"])
    with kpi3:
        st.metric("Model Architecture", "MobileNetV2")
    with kpi4:
        if test_metrics:
            st.metric("Test Accuracy", f"{test_metrics.get('test_accuracy', 'N/A')}%")
        else:
            st.metric("Test Accuracy", "Not Evaluated")

    st.markdown("---")

    # Metrics Details Section
    if test_metrics:
        st.markdown("### 🎯 Verified Test Evaluation Metrics")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Overall Accuracy", f"{test_metrics.get('test_accuracy')}%")
        m_col2.metric("Macro Precision", f"{test_metrics.get('macro_precision')}%")
        m_col3.metric("Macro Recall", f"{test_metrics.get('macro_recall')}%")
        m_col4.metric("Macro F1-Score", f"{test_metrics.get('macro_f1')}%")
        st.caption(f"Evaluated on {test_metrics.get('total_test_samples', 0)} unseen test set images.")
    else:
        st.info(
            "ℹ️ **Evaluation Status:** The model has not been evaluated on test data yet. "
            "To generate real test accuracy, precision, recall, and confusion matrix charts, run: "
            "`python evaluate_model.py` in your terminal."
        )

    st.markdown("### 📈 Visual Evaluation Charts")
    chart_tabs = st.tabs([
        "Training & Validation Curves",
        "Confusion Matrix",
        "Class Performance",
        "Dataset Distribution"
    ])

    with chart_tabs[0]:
        acc_path = os.path.join("results", "accuracy.png")
        loss_path = os.path.join("results", "loss.png")
        if os.path.isfile(acc_path) and os.path.isfile(loss_path):
            c1, c2 = st.columns(2)
            with c1:
                st.image(acc_path, caption="Accuracy Curve (Train vs Validation)", use_container_width=True)
            with c2:
                st.image(loss_path, caption="Loss Curve (Train vs Validation)", use_container_width=True)
        else:
            st.warning("Training curves will appear here after running `python train_model.py`.")

    with chart_tabs[1]:
        cm_path = os.path.join("results", "confusion_matrix.png")
        if os.path.isfile(cm_path):
            st.image(cm_path, caption="Confusion Matrix on Test Dataset", use_container_width=True)
        else:
            st.warning("Confusion matrix will appear here after running `python evaluate_model.py`.")

    with chart_tabs[2]:
        perf_path = os.path.join("results", "class_performance.png")
        if os.path.isfile(perf_path):
            st.image(perf_path, caption="Class-wise F1-Score Breakdown", use_container_width=True)
        else:
            st.warning("Class performance chart will appear here after running `python evaluate_model.py`.")

    with chart_tabs[3]:
        dist_path = os.path.join("results", "dataset_distribution.png")
        if os.path.isfile(dist_path):
            st.image(dist_path, caption="Dataset Split Distribution (Train / Val / Test)", use_container_width=True)
        else:
            st.info("Run `python preprocess_data.py --source_dir <path>` to generate the dataset distribution chart.")

    # Class Knowledge Catalog Explorer
    st.markdown("---")
    st.markdown("### 📚 Crop & Disease Catalog Explorer")
    selected_crop = st.selectbox("Filter conditions by crop:", ["All"] + db_stats["crops_list"])

    if selected_crop == "All":
        display_crops = db_stats["crops_list"]
    else:
        display_crops = [selected_crop]

    catalog_data = []
    for c in display_crops:
        for d in get_diseases_for_crop(c):
            catalog_data.append({
                "Crop": d["crop"],
                "Condition": d["disease"],
                "Status": d["status"],
                "Pathogen": d.get("pathogen_type", "N/A")
            })

    st.dataframe(pd.DataFrame(catalog_data), use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE 4: ABOUT PROJECT
# -----------------------------------------------------------------------------
def render_about_page():
    st.markdown("""
    <div class="hero-banner">
        <h1>Project Documentation</h1>
        <p>BCA Third-Year Summer Project: Plant Disease Detection Using AI/ML</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 📌 Project Synopsis
    - **Project Title:** Plant Disease Detection Using AI/ML
    - **Academic Level:** Bachelor of Computer Applications (BCA) - 3rd Year Summer Project
    - **Domain:** Artificial Intelligence, Computer Vision, Precision Agriculture
    - **Core Technologies:** Python, TensorFlow, Keras, MobileNetV2, OpenCV, Streamlit, Scikit-learn
    - **Inference Mode:** 100% Offline Local Neural Network Inference
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 1. Problem Statement")
        st.markdown("""
        Plant diseases account for **20% to 40% of global crop yield losses annually**. 
        Traditional manual diagnosis relies on agricultural extension specialists physically inspecting leaves, 
        which is slow, subjective, and inaccessible in remote rural farming regions. 
        Delayed diagnosis frequently leads to misapplication of costly, hazardous chemicals or outright harvest failure.
        """)

        st.markdown("#### 2. Project Objectives")
        st.markdown("""
        1. Build an automated visual classification pipeline using Convolutional Neural Networks.
        2. Leverage Transfer Learning (MobileNetV2) to enable high diagnostic accuracy on consumer laptops.
        3. Provide top-3 probability confidence rankings rather than blind single predictions.
        4. Deliver structured agricultural advisories detailing symptoms, prevention, and safe management.
        5. Provide a clean, beginner-friendly Streamlit web interface suitable for university viva presentation.
        """)

    with col2:
        st.markdown("#### 3. Deep Learning Methodology")
        st.markdown("""
        - **Feature Extractor:** MobileNetV2 pretrained on ImageNet (1.4 million images).
        - **Depthwise Separable Convolutions:** Divides standard convolution into depthwise and pointwise stages, reducing parameters by ~8-9x without loss of accuracy.
        - **Regularization:** Batch Normalization and 40% Dropout to curb overfitting.
        - **Data Augmentation:** Real-time botanical rotations, flips, shifts, and zoom to improve generalization.
        - **Loss Function:** Categorical Cross-Entropy optimized with the Adam algorithm.
        """)

        st.markdown("#### 4. Technology Stack")
        st.markdown("""
        | Layer | Technology |
        |---|---|
        | Language | Python 3.9 - 3.11 |
        | Deep Learning | TensorFlow 2.15+, Keras |
        | Pretrained Model | MobileNetV2 (ImageNet) |
        | Image Processing | OpenCV, Pillow |
        | Metrics & Splits | Scikit-learn |
        | Visualization | Matplotlib, Seaborn |
        | Web Application | Streamlit Framework |
        """)

    st.markdown("---")

    st.markdown("#### 5. Advantages & Practical Benefits")
    st.markdown("""
    - **High Computational Efficiency:** MobileNetV2 weighs only ~14 MB, executing inference in under 100ms on standard CPUs without requiring expensive GPUs.
    - **Transparency & Trust:** Transparent top-3 probability breakdown prevents overconfidence on ambiguous images.
    - **Decision Support:** Connects raw classification labels directly with actionable farming guidelines and expert disclaimers.
    - **Zero Data Leakage:** Stratified hash-checked dataset splitting safeguards evaluation integrity.
    """)

    st.markdown("#### 6. Limitations & Future Scope")
    st.markdown("""
    - **Current Limitations:** Classification accuracy depends on clear single-leaf images with uniform illumination. Extreme camera blur or multi-leaf occlusion can degrade confidence.
    - **Future Scope:**
      1. Edge deployment to Android/iOS smartphones via TensorFlow Lite (TFLite).
      2. Autonomous drone scouting with YOLOv8 for real-time field-level object detection.
      3. Integration with IoT soil moisture and atmospheric humidity sensors for multi-modal risk forecasting.
      4. Multilingual voice synthesis in regional languages (e.g. Hindi, Marathi) for rural accessibility.
    """)


# -----------------------------------------------------------------------------
# MAIN APP ROUTER
# -----------------------------------------------------------------------------
def main():
    nav_choice = render_sidebar()

    if nav_choice == "🌿 Home":
        render_home_page()
    elif nav_choice == "🔍 Disease Detection":
        render_detection_page()
    elif nav_choice == "📊 Dashboard":
        render_dashboard_page()
    elif nav_choice == "📖 About Project":
        render_about_page()


if __name__ == "__main__":
    main()
