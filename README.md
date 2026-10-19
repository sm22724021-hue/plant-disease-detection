# Plant Health Advisor
### Local leaf-image screening and crop health information

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
An offline image-based plant health screening application. It runs a trained image classifier locally and connects supported predictions with crop and disease information. The model can recognize only the conditions represented by labeled images used during training; catalog information for other crops does not mean the model can diagnose them.

---

## Table of Contents
1. [Application Overview](#1-application-overview)
2. [Key Features](#2-key-features)
3. [Supported Crops & Pathologies](#3-supported-crops--pathologies)
4. [Technology Stack](#4-technology-stack)
5. [System Requirements](#5-system-requirements)
6. [Application Architecture](#6-application-architecture)
7. [Installation & Setup](#7-installation--setup)
8. [Dataset Acquisition & Preparation](#8-dataset-acquisition--preparation)
9. [Model Training](#9-model-training)
10. [Model Evaluation & Metrics](#10-model-evaluation--metrics)
11. [Running the Web Application](#11-running-the-web-application)
12. [How the AI Pipeline Works](#12-how-the-ai-pipeline-works)
13. [How to Add New Crops & Disease Classes](#13-how-to-add-new-crops--disease-classes)
14. [Troubleshooting & FAQs](#14-troubleshooting--faqs)
15. [Model Coverage and Safe Use](#15-model-coverage-and-safe-use)

---

## 1. Application Overview

The application accepts a leaf photo from file upload or camera capture and compares it with the classes in the active model. It is a screening aid, not a confirmed diagnosis, and its coverage depends on the training data installed with the model.

The active model combines the PlantVillage color dataset with a CC BY 4.0 chilli-leaf dataset: 43 labeled conditions across 15 crop names. It includes hot chilli with five classes, as well as bell pepper. The combined model scored 92.49% overall on a 10,788-image held-out test set, but chilli-specific scores vary and have small test counts (10-29 images per chilli class). Chilli results are limited screening only, not confirmed diagnoses; field photographs and local varieties may differ from the datasets.

---

## 2. Key Features

- **Language Choice:** First-open language dialog with English, Hindi, Marathi, Gujarati, and Bengali UI text; users can change it from the sidebar.
- **Genuine Neural Network Inference:** Real deep learning model execution; zero hardcoded responses or dummy mocks.
- **Lightweight Architecture:** Utilizes MobileNetV2 (~14 MB model footprint) for sub-second CPU inference on ordinary laptops.
- **Dynamic Class Adaptation:** Class labels are dynamically inferred from the dataset directory; no manual class arrays required.
- **Data Leakage Prevention:** Deduplication and stratified 70% Train / 15% Validation / 15% Test splitting via MD5 content hashing.
- **Decision Support & Safety:** Provides structured agronomic symptoms, biological prevention, and explicit expert chemical disclaimers.
- **Analytical Dashboard:** Displays dataset distributions, training/validation loss and accuracy curves, confusion matrices, and F1-score breakdowns.
- **Offline & Private:** All inferences process locally without uploading images to cloud servers.

---

## 3. Current Model Coverage

The active model recognizes 43 classes across these crops:

| Crop | Supported Conditions & Diseases |
|---|---|
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Blueberry** | Healthy |
| **Cherry** | Powdery Mildew, Healthy |
| **Chilli / Hot pepper** | Cercospora, Murda Complex (mites/thrips), Nutritional Stress, Powdery Mildew, Healthy |
| **Corn (Maize)** | Cercospora Leaf Spot / Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| **Grape** | Black Rot, Esca, Leaf Blight, Healthy |
| **Orange** | Huanglongbing (Citrus Greening) |
| **Peach** | Bacterial Spot, Healthy |
| **Pepper (Bell)** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Raspberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery Mildew |
| **Strawberry** | Leaf Scorch, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

The chilli dataset has 528 images. In the held-out test data, class F1 scores were: Cercospora 81.63% (23 images), Murda Complex 70.97% (16), Nutritional Stress 62.07% (15), Powdery Mildew 96.55% (15), and Healthy 70.00% (10). These results are limited screening evidence, not field-level guarantees; confirm suspected conditions with a local crop adviser.

The model combines the [PlantVillage color dataset](https://huggingface.co/datasets/mohanty/PlantVillage) (CC BY-SA 3.0) and the [Chilli Leaf Disease Classification dataset](https://github.com/ai-agriculture-circuits-and-systems/chilli_leaf_classification) (CC BY 4.0). The Chilli dataset includes its original Mendeley Data split files, but those lists do not match the published class folders. This application creates a deterministic 70/15/15 split and groups near-duplicate images together before splitting.

The reference catalogue includes additional conditions; a catalogue entry does not mean the model can diagnose it.

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
- **Storage:** At least 5 GB free disk space for the downloaded PlantVillage archive, prepared images, and model artifacts
- **GPU:** Optional. Model training executes on standard multi-core CPUs; NVIDIA CUDA GPU accelerates training if available.

---

## 6. Application Architecture

```
plant-disease-detection/
│
├── app.py                      # Main Streamlit web application
├── train_model.py              # End-to-end model training script
├── evaluate_model.py           # Test set evaluation and confusion matrix generator
├── preprocess_data.py          # Dataset cleaner, deduplicator & 70/15/15 splitter
├── download_dataset.py         # Dataset setup helper and verification script
├── requirements.txt            # Pinned, tested Python package dependencies
├── README.md                   # Application and model coverage guide
│
├── models_plantvillage_chilli/ # Active 43-class model and label mapping
├── models_plantvillage/        # Previous 38-class model retained as backup
├── models/                     # Previous three-class model retained as backup
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
├── results_plantvillage_chilli/ # Active model charts and evaluation reports
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
Creating a virtual environment isolates application dependencies:

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

The training scripts can use the open-access **PlantVillage** dataset (approximately 54,305 leaf images across 38 classes).

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

### Option C: Prepare the full PlantVillage color dataset used by the active model

The PlantVillage color archive is about 2.2 GB. The commands below use the official color train/test lists; the preparation script holds out the official test list and takes 10% of the training list for validation.

```bash
mkdir -p raw_dataset/plantvillage_metadata
curl -L --fail --retry 5 --retry-all-errors --continue-at - \\
   --output raw_dataset/plantvillage-data.zip \\
   https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/data.zip
curl -L --fail --output raw_dataset/plantvillage_metadata/color_train.txt \\
   https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/splits/color_train.txt
curl -L --fail --output raw_dataset/plantvillage_metadata/color_test.txt \\
   https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/splits/color_test.txt
python prepare_plantvillage.py
```

The source dataset is licensed CC-BY-SA-3.0. Check its dataset card for attribution and redistribution terms.

### Option D: Add hot chilli / hot pepper images

The app's active model combines PlantVillage with the CC BY 4.0 [Chilli Leaf Disease Classification dataset](https://github.com/ai-agriculture-circuits-and-systems/chilli_leaf_classification). The checked-in class-specific split lists do not match several image folders, so `prepare_chilli_dataset.py` creates deterministic 70/15/15 splits while grouping perceptual near-duplicates together.

```bash
mkdir -p raw_dataset/chilli_source
curl -L --fail --retry 4 --continue-at - \\
   --output raw_dataset/chilli_source/chilli_leaf_classification.zip \\
   https://github.com/ai-agriculture-circuits-and-systems/chilli_leaf_classification/archive/refs/heads/main.zip
unzip -oq raw_dataset/chilli_source/chilli_leaf_classification.zip \\
   '*/chillies/*/images/*' '*/chillies/*/sets/train.txt' \\
   '*/chillies/*/sets/val.txt' '*/chillies/*/sets/test.txt' '*/README.md' '*/LICENSE' \\
   -d raw_dataset/chilli_source/extracted
python prepare_chilli_dataset.py
python train_model.py --dataset_dir dataset_plantvillage_chilli_v2 --models_dir models_plantvillage_chilli --results_dir results_plantvillage_chilli --epochs 10 --batch_size 32
```

The chilli dataset contains 528 images across five classes and provides small held-out samples per class. The model therefore reports a chilli screening result, not a definitive diagnosis.

---

## 9. Model Training

Train the deep learning model using:
```bash
python train_model.py --dataset_dir dataset_plantvillage_chilli_v2 --models_dir models_plantvillage_chilli --results_dir results_plantvillage_chilli --epochs 10 --batch_size 32
```

### What happens during training:
1. **Dynamic Class Scanning:** Reads all class folders in the selected training directory and saves a matching class mapping.
2. **Data Pipeline Optimization:** Loads images via `tf.keras.utils.image_dataset_from_directory` with GPU/CPU `AUTOTUNE` prefetching.
3. **Biological Augmentation:** Applies random flips, gentle rotations ($\pm 15^\circ$), zoom ($\pm 15\%$), and contrast variations to prevent overfitting.
4. **Transfer Learning Initialization:** Loads MobileNetV2 pretrained on ImageNet; freezes the base layers; initializes Global Average Pooling, Batch Normalization, Dense(256), and Dropout(0.4).
5. **Smart Callbacks:**
   - `EarlyStopping`: Halts training if validation loss plateaus for 5 epochs.
   - `ModelCheckpoint`: Automatically saves the best model checkpoint to the selected model directory.
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
- **🔍 Disease Detection:** Select a crop, then upload one or more JPG/PNG images, capture a photo, or choose a sample. Analyze the batch to view separate candidate rankings and reference information for each image.
- **📊 Dashboard:** Live KPI cards, test accuracy, interactive training curves, confusion matrix, and a searchable crop-disease encyclopedia.
- **ℹ️ About the Tool:** Learned model classes, coverage limits, and safe-use guidance.

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

1. Add a new crop/disease class with a large, varied set of correctly labeled images in the `train`, `validation`, and `test` splits following the naming convention:
   `CropName___Disease_Name` (e.g., `Mango___Anthracnose` or `Rice___Blast`).
2. Keep images of the same physical leaf out of both training and test splits. One or a few images are not enough for reliable field use.
3. Run `python train_model.py --dataset_dir <dataset_root> --models_dir <model_output> --results_dir <results_output> --epochs 10`.
4. The system **automatically**:
   - Discovers the new class.
   - Updates the selected output directory's `class_names.json`.
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

## 15. Model Coverage and Safe Use

- Predictions are available only for the classes listed in `models_plantvillage_chilli/class_names.json`.
- The model includes hot chilli classes, but their held-out test sets are small and performance varies substantially by class. Treat chilli outputs as an initial screen only.
- The disease reference catalogue can provide information for crops beyond the classifier's coverage; it does not expand what the model can predict.
- Image scores are screening signals, not confirmed diagnoses. Verify symptoms and consult a qualified local agricultural adviser before applying treatment.
- The main navigation, crop selection, upload flow, result labels, and safety notices have interface translations. Detailed disease-reference descriptions and some dataset disease names remain in English; they should be professionally reviewed before relying on them in another language.
# plant-disease-detection
