# 🌾 CropVision: Agricultural Crop Plant Image Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![Classes](https://img.shields.io/badge/Crop%20Classes-139-green.svg)](#dataset)

An end-to-end Deep Learning & Computer Vision project designed to classify **139 distinct agricultural crop plant species** from leaf and plant field images. The project includes a complete PyTorch training and evaluation pipeline, transfer learning backbones, an interactive Streamlit web dashboard with agronomic farming profiles, and a well-annotated Jupyter Notebook.

---

## 🌾 Project Overview

Accurate identification of agricultural crops is fundamental for precision farming, automated weeding, yield estimation, and crop biodiversity tracking. 

This project solves this challenge by implementing:
- **Transfer Learning Backbone:** ResNet-18 (pre-trained on ImageNet) fine-tuned for 139 agricultural crop classes.
- **Data Augmentation:** Random rotations, horizontal flips, color jitter, and standard ImageNet normalization to increase model robustness against varying natural light and field conditions.
- **Top-1 and Top-5 Evaluation:** Multi-class evaluation metrics capturing true species retrieval confidence.
- **Interactive Web App (Streamlit):** Web dashboard allowing users to upload field photos, view Top-5 prediction probabilities, and read crop-specific cultivation guidelines (soil, water, climate).

---

## 📊 Dataset Structure

The dataset contains RGB images standardly resized to `224x224` pixels across 139 agricultural plant categories (including Rice, Wheat, Maize, Cotton, Coffee, Bananas, Tea, Apples, Potatoes, Chickpeas, and more).

```
RGB_224x224/
├── train/       # 27,639 images across 139 classes
├── val/         # 6,422 images across 139 classes
└── test/        # 6,627 images across 139 classes
```

---

## 🏗 Architecture & Pipeline

```
[ Input Crop Image (224x224) ]
               │
               ▼
   [ Data Preprocessing & Augmentation ]
     • RandomHorizontalFlip(p=0.5)
     • RandomRotation(±15°)
     • ColorJitter(brightness=0.2, contrast=0.2)
     • Normalization (ImageNet stats)
               │
               ▼
   [ Pre-trained ResNet-18 Feature Extractor ]
     • Residual Blocks (Conv layers with skip connections)
     • Global Average Pooling (512-dim embedding)
               │
               ▼
   [ Adapted Classification Head ]
     • Dropout (p=0.2)
     • Linear(in_features=512, out_features=139)
               │
               ▼
   [ Softmax Probabilities & Top-K Predictions ]
     • Top-1 Prediction & Confidence %
     • Top-5 Class Candidates
     • Agronomic Cultivation Insights
```

---

## 📂 Project Directory Layout

```
Crop image classification/
├── RGB_224x224/                # Image splits: train, val, and test
│   ├── train/
│   ├── val/
│   └── test/
├── models/                     # Saved checkpoints & class index mappings
│   ├── class_names.json        # 139 crop class dictionary
│   ├── crop_resnet18_best.pth  # Best trained model weights
│   ├── training_history.json   # Epoch loss & accuracy logs
│   └── evaluation_results.json # Test metrics report
├── src/                        # Modular Python Package
│   ├── __init__.py
│   ├── dataset.py              # Transforms and DataLoaders
│   ├── model.py                # ResNet/MobileNet model definitions
│   ├── train.py                # Training & validation loop
│   ├── evaluate.py             # Test split evaluation
│   └── predict.py              # Single/batch inference module
├── app.py                      # Interactive Streamlit Web App
├── model_training.ipynb        # Clean Jupyter Notebook walkthrough
├── requirements.txt            # Dependency list
└── README.md                   # Documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone / Navigate to Project Directory
```powershell
cd "c:\Users\princ\OneDrive\Desktop\New folder\Crop image classification"
```

### 2. Activate Virtual Environment
```powershell
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🚀 Training & Evaluation

### Training the Model
To train the classifier on the dataset:
```powershell
# Standard training (5 epochs, batch size 32):
python -m src.train --epochs 5 --batch-size 32 --lr 0.001 --architecture resnet18

# Rapid prototype on CPU (using a balanced subset of 10 samples per class):
python -m src.train --epochs 3 --batch-size 32 --samples-per-class 10
```

### Evaluating on Test Set
Evaluate the trained model on the `RGB_224x224/test` dataset:
```powershell
python -m src.evaluate --checkpoint models/crop_resnet18_best.pth
```
Outputs:
- **Top-1 Accuracy (%)**
- **Top-5 Accuracy (%)**
- **Per-class performance breakdown**

---

## 🖥️ Running the Interactive Web App

Launch the Streamlit web dashboard:
```powershell
streamlit run app.py
```

### Features:
1. **🔍 Crop Identifier:**
   - Upload any plant/leaf photo, pick from test dataset gallery, or take a picture with a webcam.
   - Real-time Top-1 classification and Top-5 probability distribution.
   - Agronomic cultivation guidance (ideal climate, soil, water requirement, farming tips).
2. **📊 Model Performance:**
   - Training vs Validation loss curves.
   - Top-1 and Top-5 accuracy trajectories.
3. **🗂️ Dataset Explorer:**
   - Search and browse through all 139 supported crop plant species.

---

## 🔮 CLI Prediction API

Classify any crop image directly from the terminal:
```powershell
python -m src.predict --image "RGB_224x224\test\Maize (Corn) plant\sample.jpg" --top-k 5
```

Sample output:
```text
==================================================
 Predictions for: sample.jpg
==================================================
#01 | Maize (Corn)                   |  84.20% | ████████████████
#02 | Sorghum                        |   6.15% | █
#03 | Millet                         |   3.40% | 
#04 | Barley                         |   2.10% | 
#05 | Wheat                          |   1.20% | 
==================================================
```

---

## 📓 Jupyter Notebook

The repository includes a fully commented and debugged notebook:
[`model_training.ipynb`](model_training.ipynb)
- Section 1: Setup & Libraries
- Section 2: Exploratory Data Analysis & Class Distribution
- Section 3: Data Augmentation & Loaders
- Section 4: Transfer Learning Model Architecture
- Section 5: Optimization & Training Loop
- Section 6: Epoch Execution & Checkpointing
- Section 7: Evaluation & Prediction Visualizer
- Section 8: Model Serialization
