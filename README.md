# Plant Disease Detection Using AI/ML
### BCA Third-Year Summer Project | University Examination & Viva Ready

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Academic%20Use-green.svg)]()

An end-to-end, production-grade Artificial Intelligence and Deep Learning system designed to identify and classify plant foliar diseases from leaf images. The application leverages **Transfer Learning (MobileNetV2)** with ImageNet pretrained feature representations, depthwise separable convolutions, automated stratified dataset splitting, real-time probability confidence ranking, and an agricultural advisory knowledge base.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [Supported Crops & Pathologies](#3-supported-crops--pathologies)
4. [Technology Stack](#4-technology-stack)
5. [System Requirements](#5-system-requirements)
6. [Project Architecture](#6-project-architecture)
7. [Installation & Setup](#7-installation--setup)
8. [Dataset Acquisition & Preparation](#8-dataset-acquisition--preparation)
9. [Model Training](#9-model-training)
10. [Model Evaluation & Metrics](#10-model-evaluation--metrics)
11. [Running the Web Application](#11-running-the-web-application)
12. [How the AI Pipeline Works](#12-how-the-ai-pipeline-works)
13. [How to Add New Crops & Disease Classes](#13-how-to-add-new-crops--disease-classes)
14. [Troubleshooting & FAQs](#14-troubleshooting--faqs)
15. [University Viva Defense Guide (15 Questions & Answers)](#15-university-viva-defense-guide)

---

## 1. Project Overview

Plant pathologies threaten between 20% to 40% of global agricultural yields every year. Traditional agronomic diagnosis relies on manual visual examination by scarce agricultural extension specialists, leading to delayed diagnoses, misapplication of chemical pesticides, and substantial financial losses for smallholder farmers.

This project delivers a complete, local machine learning solution that:
- Accepts leaf images via file upload or camera capture.
- Analyzes leaf tissue using deep convolutional representations.
- Predicts whether the specimen is **Healthy** or **Diseased**.
- Identifies the specific fungal, bacterial, or viral disease class.
- Outputs top-3 prediction probabilities with calibrated confidence scores.
- Provides actionable symptoms, prevention tips, and safe management protocols.

---

## 2. Key Features

- **Genuine Neural Network Inference:** Real deep learning model execution; zero hardcoded responses or dummy mocks.
- **Lightweight Architecture:** Utilizes MobileNetV2 (~14 MB model footprint) for sub-second CPU inference on ordinary laptops.
- **Dynamic Class Adaptation:** Class labels are dynamically inferred from the dataset directory; no manual class arrays required.
- **Data Leakage Prevention:** Deduplication and stratified 70% Train / 15% Validation / 15% Test splitting via MD5 content hashing.
- **Decision Support & Safety:** Provides structured agronomic symptoms, biological prevention, and explicit expert chemical disclaimers.
- **Analytical Dashboard:** Displays dataset distributions, training/validation loss and accuracy curves, confusion matrices, and F1-score breakdowns.
- **Offline & Private:** All inferences process locally without uploading images to cloud servers.

---

## 3. Supported Crops & Pathologies

The system ships with pre-configured agronomic profiles for major agricultural crops:

| Crop | Supported Conditions & Diseases |
|---|---|
| **Tomato** | Early Blight, Late Blight, Bacterial Spot, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Tomato Yellow Leaf Curl Virus (TYLCV), Tomato Mosaic Virus (ToMV), Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Pepper (Bell)** | Bacterial Spot, Healthy |
| **Maize (Corn)** | Gray Leaf Spot (*Cercospora*), Common Rust, Northern Leaf Blight, Healthy |
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Grape** | Black Rot, Esca (*Black Measles*), Leaf Blight (*Isariopsis*), Healthy |

*Note: New crops and disease classes can be added seamlessly by placing image folders into `dataset/train/` without editing code.*

---

## 4. Technology Stack

- **Programming Language:** Python 3.9, 3.10, or 3.11
- **Deep Learning Framework:** TensorFlow 2.15+ / Keras
- **Core Architecture:** MobileNetV2 (ImageNet Transfer Learning)
- **Computer Vision:** OpenCV (`opencv-python`), Pillow (`PIL`)
- **Data Processing:** NumPy, Pandas, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Web Interface:** Streamlit Framework

---

## 5. System Requirements

- **Operating System:** Windows 10/11, macOS (Intel or Apple Silicon M1/M2/M3), or Linux (Ubuntu 20.04+)
- **Processor (CPU):** Intel Core i3/i5/i7 (8th Gen+) or AMD Ryzen 3/5/7, or Apple Silicon
- **RAM:** 8 GB minimum (16 GB recommended for model training)
- **Storage:** At least 3 GB free disk space (accounting for dataset and model weights)
- **GPU:** Optional. Model training executes on standard multi-core CPUs; NVIDIA CUDA GPU accelerates training if available.

---

## 6. Project Architecture

```
plant-disease-detection/
│
├── app.py                      # Main Streamlit web application
├── train_model.py              # End-to-end model training script
├── evaluate_model.py           # Test set evaluation and confusion matrix generator
├── preprocess_data.py          # Dataset cleaner, deduplicator & 70/15/15 splitter
├── download_dataset.py         # Dataset setup helper and verification script
├── requirements.txt            # Pinned, tested Python package dependencies
├── README.md                   # Comprehensive project documentation
│
├── models/                     # Saved models and class labels
│   ├── plant_disease_model.keras  # Best trained MobileNetV2 model weights
│   └── class_names.json        # Dynamic index-to-class mapping
│
├── dataset/                    # Dataset directory structure
│   ├── train/                  # 70% Training images (class subfolders)
│   ├── validation/             # 15% Validation images
│   └── test/                   # 15% Unseen test evaluation images
│
├── src/                        # Modular source package
│   ├── __init__.py             # Package marker
│   ├── preprocessing.py        # Image validation, resizing, normalization, augmentation
│   ├── prediction.py           # Cached inference engine, top-K ranking
│   ├── model.py                # MobileNetV2 and custom CNN architectures
│   └── disease_info.py         # Agronomic knowledge base & dynamic parser
│
├── results/                    # Generated charts & analytical reports
│   ├── accuracy.png            # Training vs Validation accuracy curve
│   ├── loss.png                # Training vs Validation loss curve
│   ├── confusion_matrix.png    # Test set confusion matrix heatmap
│   ├── class_performance.png   # Class-wise F1-score horizontal bar chart
│   ├── dataset_distribution.png# Split distribution bar chart
│   ├── classification_report.txt # Raw precision, recall, F1 text report
│   └── metrics.json            # Structured metrics for Streamlit dashboard
│
├── samples/                    # Pre-packaged sample leaves for quick demo
│   ├── tomato_early_blight.jpg
│   ├── tomato_healthy.jpg
│   └── potato_late_blight.jpg
│
└── assets/
    └── logo.png                # System vector/branding logo
```

---

## 7. Installation & Setup

### Step 1: Open Terminal / Command Prompt
Navigate to your desired workspace directory:
```bash
cd plant-disease-detection
```

### Step 2: Create a Python Virtual Environment
Creating a virtual environment isolates project dependencies:

- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

- **Windows (Command Prompt):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate
  ```

- **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

*(If on macOS Xcode developer tools are requested, run `xcode-select --install` first).*

### Step 3: Install Required Dependencies
Install the verified dependencies from `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 8. Dataset Acquisition & Preparation

The project is designed around the open-access **PlantVillage** dataset (~54,305 leaf images across 38 classes).

### Option A: Manual Download from Kaggle (Recommended)
1. Download the dataset zip from Kaggle:
   👉 **URL:** [https://www.kaggle.com/datasets/emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease)
2. Extract the zip file to a folder named `raw_dataset/`:
   ```
   plant-disease-detection/
   └── raw_dataset/
       ├── Tomato___Early_blight/
       ├── Tomato___healthy/
       ├── Potato___Late_blight/
       └── ...
   ```
3. Run the automated data leakage prevention & splitting script:
   ```bash
   python preprocess_data.py --source_dir raw_dataset
   ```
   This script:
   - Removes corrupted and zero-byte files.
   - Computes MD5 file hashes to eliminate duplicates.
   - Performs a stratified 70% Train, 15% Validation, and 15% Test split.
   - Saves a visual distribution chart to `results/dataset_distribution.png`.

### Option B: Using the Download Helper
Run:
```bash
python download_dataset.py --verify
```

---

## 9. Model Training

Train the deep learning model using:
```bash
python train_model.py --epochs 15 --batch_size 32 --lr 0.001
```

### What happens during training:
1. **Dynamic Class Scanning:** Reads all class folders in `dataset/train/` and saves `models/class_names.json`.
2. **Data Pipeline Optimization:** Loads images via `tf.keras.utils.image_dataset_from_directory` with GPU/CPU `AUTOTUNE` prefetching.
3. **Biological Augmentation:** Applies random flips, gentle rotations ($\pm 15^\circ$), zoom ($\pm 15\%$), and contrast variations to prevent overfitting.
4. **Transfer Learning Initialization:** Loads MobileNetV2 pretrained on ImageNet; freezes the base layers; initializes Global Average Pooling, Batch Normalization, Dense(256), and Dropout(0.4).
5. **Smart Callbacks:**
   - `EarlyStopping`: Halts training if validation loss plateaus for 5 epochs.
   - `ModelCheckpoint`: Automatically saves the best model checkpoint to `models/plant_disease_model.keras`.
   - `ReduceLROnPlateau`: Halves learning rate when progress stalls.
6. **Report Generation:** Generates and saves `results/accuracy.png` and `results/loss.png`.

---

## 10. Model Evaluation & Metrics

Evaluate the trained model on completely unseen test images:
```bash
python evaluate_model.py
```

### Generated Deliverables:
- **`results/classification_report.txt`**: Precision, Recall, F1-Score, and Support for each class.
- **`results/metrics.json`**: Machine-readable metrics loaded by the Streamlit dashboard.
- **`results/confusion_matrix.png`**: High-resolution heatmap showing true vs predicted class distributions.
- **`results/class_performance.png`**: Horizontal bar chart comparing F1-scores across all conditions.

---

## 11. Running the Web Application

Launch the Streamlit web application with:
```bash
streamlit run app.py
```

The application will launch in your default browser at:
`http://localhost:8501`

### Exploring the Application:
- **🌿 Home:** Executive overview, biological workflow, supported crops, and offline privacy statement.
- **🔍 Disease Detection:** Upload a JPG/PNG, capture live via webcam, or choose a sample image. Click **Detect Disease** to view:
  - Diagnosis Card (Crop, Disease Name, Status Pill).
  - Confidence Score and interactive visual meter.
  - Top-3 candidate ranking with probabilities.
  - Detailed agronomic advisory tabs (Overview, Visible Symptoms, Prevention, Field Management, and Agricultural Disclaimer).
- **📊 Dashboard:** Live KPI cards, test accuracy, interactive training curves, confusion matrix, and a searchable crop-disease encyclopedia.
- **📖 About Project:** BCA summer project background, problem statement, objectives, architecture diagram, and viva preparation guide.

---

## 12. How the AI Pipeline Works

```
                                [ Input Leaf Image ]
                                         │
                                         ▼
                             [ Image Preprocessing ]
                     (Resize 224x224, RGB Mode, /255.0 Scale)
                                         │
                                         ▼
                            [ MobileNetV2 Base Model ]
                      (Depthwise Separable Convolutions)
                                         │
                                         ▼
                          [ Global Average Pooling 2D ]
                                         │
                                         ▼
                             [ Batch Normalization ]
                                         │
                                         ▼
                            [ Dense Layer (256, ReLU) ]
                                         │
                                         ▼
                               [ Dropout Layer (0.4) ]
                                         │
                                         ▼
                          [ Softmax Output (N Classes) ]
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
         [ Top-1 Prediction & ]                     [ Agronomic Knowledge ]
        [ Top-3 Probabilities ]                     [   Base Association  ]
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                            [ Web Application Display ]
```

---

## 13. How to Add New Crops & Disease Classes

1. Create a new folder inside `dataset/train/`, `dataset/validation/`, and `dataset/test/` following the naming convention:
   `CropName___Disease_Name` (e.g., `Mango___Anthracnose` or `Rice___Blast`).
2. Add leaf photographs of that disease into the folder.
3. Run `python train_model.py --epochs 15`.
4. The system **automatically**:
   - Discovers the new class.
   - Updates `models/class_names.json`.
   - Adjusts the neural network output layer.
   - Parses the crop and disease names in the UI via the dynamic fallback parser in `src/disease_info.py`.

---

## 14. Troubleshooting & FAQs

### Q1: "ModuleNotFoundError: No module named 'tensorflow'"
**Solution:** Ensure your virtual environment is activated before running scripts:
```bash
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
pip install -r requirements.txt
```

### Q2: "Model not trained yet. Please run train_model.py"
**Solution:** The web application requires the trained weights file. Run `python train_model.py` to train the model on your dataset, or place a trained `.keras` file into the `models/` directory.

### Q3: "Unable to process this image. The uploaded image is too small..."
**Solution:** The image validation layer requires images to be at least 32x32 pixels to prevent corrupted or blank inputs from crashing the neural network. Upload a clear, focused leaf photograph.

### Q4: "TensorFlow warning: CPU does not support AVX2 instructions"
**Solution:** This is an informational warning and does not prevent execution. The model runs smoothly on standard CPUs.

---

## 15. University Viva Defense Guide

Here are **15 likely questions** asked by university external examiners during BCA project vivas, along with standard technical answers:

#### Q1: What is the main objective of this project?
**Answer:** The objective is to build an end-to-end, automated agricultural decision-support tool that uses computer vision and deep learning to identify plant foliar diseases early from leaf photographs, assisting farmers in applying targeted treatments before widespread crop damage occurs.

#### Q2: Why did you choose MobileNetV2 instead of a traditional CNN from scratch?
**Answer:** MobileNetV2 is pretrained on the 1.4-million-image ImageNet dataset, meaning its lower layers already understand fundamental visual primitives like edges, textures, and color gradients. By using Transfer Learning, we achieve higher accuracy with significantly less training time and fewer epochs. Additionally, MobileNetV2 uses depthwise separable convolutions, resulting in a small model (~14 MB) capable of running fast inference on low-cost consumer hardware without requiring an expensive GPU.

#### Q3: What are Depthwise Separable Convolutions?
**Answer:** Standard convolution performs spatial filtering and channel combining in a single step, which is computationally expensive. Depthwise separable convolution splits this into two steps:
1. *Depthwise convolution:* applies a single 3x3 filter per input channel.
2. *Pointwise convolution:* uses a 1x1 convolution to linearly combine the outputs across channels.
This reduces computation and parameter count by approximately 8 to 9 times with minimal loss of accuracy.

#### Q4: How did you prevent data leakage during dataset preparation?
**Answer:** Data leakage occurs when identical or near-identical images appear in both training and test sets, artificially inflating test accuracy. In `preprocess_data.py`, we compute MD5 hashes for all images to discard duplicate files. We then perform a stratified 70% Train, 15% Validation, and 15% Test split before any training begins.

#### Q5: What is the purpose of Data Augmentation?
**Answer:** Data augmentation artificially expands the diversity of training images by applying realistic transformations like random rotations ($\pm 15^\circ$), horizontal and vertical flips, shifts, and slight contrast adjustments. This teaches the network that a leaf is still the same class regardless of its orientation or lighting, mitigating overfitting and improving real-world generalization.

#### Q6: Why did you use Categorical Cross-Entropy loss?
**Answer:** Categorical Cross-Entropy is the standard loss function for multi-class classification where each sample belongs to exactly one discrete class. It measures the divergence between the true one-hot encoded distribution and the predicted Softmax probability distribution, heavily penalizing confident wrong predictions.

#### Q7: What role does the Softmax activation function play in the output layer?
**Answer:** Softmax normalizes raw neural network outputs (logits) into a valid probability distribution where all values range between 0.0 and 1.0 and sum up to 1.0. This allows us to interpret outputs as percentage confidence scores and rank the top-3 most probable diseases.

#### Q8: What is the role of the Dropout layer?
**Answer:** Dropout is a regularization technique. During training, it randomly sets a fraction (in our case, 40%) of the hidden units' activations to zero at each update step. This prevents neurons from co-adapting excessively, forcing the network to learn robust, generalized representations.

#### Q9: What callbacks were utilized during model training?
**Answer:** We implemented three Keras callbacks:
1. `EarlyStopping`: halts training if validation loss fails to improve for 5 consecutive epochs, restoring the best historical weights.
2. `ModelCheckpoint`: automatically serializes the best performing model based on validation accuracy to `models/plant_disease_model.keras`.
3. `ReduceLROnPlateau`: lowers the learning rate by a factor of 0.3 when validation loss stagnates, allowing the optimizer to settle into narrow local minima.

#### Q10: What is the difference between Validation Accuracy and Test Accuracy?
**Answer:** Validation accuracy is monitored during training to tune hyperparameters, adjust the learning rate, and trigger early stopping. Because the model indirectly influences hyperparameter selection via the validation set, an independent **Test Set** that the model has never seen during training or tuning is evaluated using `evaluate_model.py` to obtain an unbiased measure of true generalization performance.

#### Q11: Why is Precision and Recall important alongside Accuracy?
**Answer:** In agricultural diagnosis, classes can have unequal sample counts. Overall accuracy can be misleading if a majority class dominates. Precision measures how many of the detected diseased leaves actually had the disease (avoiding false alarms), while Recall (Sensitivity) measures how many actual diseased leaves the model successfully identified (avoiding missed infections). The F1-Score represents the harmonic mean of both.

#### Q12: What is a Confusion Matrix?
**Answer:** A confusion matrix is an $N \times N$ table where rows represent true ground-truth classes and columns represent predicted classes. Diagonal elements represent correct classifications, while off-diagonal elements illustrate exact points of confusion (e.g., distinguishing Early Blight from Septoria Leaf Spot due to similar dark concentric spots).

#### Q13: How does Streamlit's `@st.cache_resource` optimize application performance?
**Answer:** Without caching, Streamlit reruns the entire Python script from top to bottom on every user interaction (such as button clicks or slider adjustments). Re-loading a multi-megabyte neural network from disk on every click would cause severe lag. `@st.cache_resource` loads the model weights once into system memory and shares the instance across all interactions.

#### Q14: What safety mechanisms are built into the disease recommendations?
**Answer:** Automated AI classification is a screening and decision-support tool. It must never recommend indiscriminate chemical application. The application incorporates a clear **Agricultural Advisory Disclaimer** advising farmers to consult certified plant pathologists or local agricultural extension officers before handling restricted chemical fungicides.

#### Q15: What are the future enhancements planned for this project?
**Answer:**
1. Exporting the trained model to TensorFlow Lite (`.tflite`) for deployment on mobile Android applications without internet connectivity.
2. Integrating drone-based aerial imagery with YOLO object detection for canopy-level field monitoring.
3. Multi-modal prediction by combining leaf images with temperature and humidity IoT sensor data.
4. Adding local language voice synthesis (Hindi, Marathi, Telugu) for illiterate farmers.
