# 🚗 AI-Powered Vehicle Damage Assessment & Repair Cost Estimator

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458.svg)

## 📌 Overview
This project is an end-to-end, multi-modal machine learning pipeline designed to automate auto insurance claims. It leverages Computer Vision to assess vehicle collision damage from unstructured images and utilizes Tabular Machine Learning to estimate localized repair costs based on the vehicle's specific make, model, and manufacturing year.

## ⚙️ Dual-Model Architecture

### 1. The Vision Model (Damage Severity Classification)
* **Architecture:** Convolutional Neural Network (CNN) via **ResNet-50** (Pre-trained on ImageNet).
* **Implementation:** The base layers are frozen for feature extraction, while a custom classification head (with Dropout for regularization) is fine-tuned to classify vehicle damage into three distinct categories: `Minor`, `Moderate`, and `Severe`.
* **Output:** A calibrated Softmax probability distribution indicating damage severity confidence.

### 2. The Pricing Engine (Financial Estimation)
* **Architecture:** **HistGradientBoostingRegressor** (Scikit-Learn's highly optimized, native alternative to XGBoost).
* **Implementation:** A structured pipeline utilizing `ColumnTransformer`, `StandardScaler`, and `OneHotEncoder` to process vehicle metadata (Make, Model, Year) alongside the predicted severity from the Vision Model.
* **Output:** A continuous numerical prediction representing the estimated repair cost in USD.

## 📁 Repository Structure
```text
├── data3a 2/                             # Image data directories (minor, moderate, severe)
├── saved_models/                         # Serialized .pth and .pkl model files
├── reports/                              # Output directory for visual receipts
├── 01_vision_model_training.ipynb        # ResNet-50 training, augmentation, and evaluation
├── 02_pricing_model_training.ipynb       # HistGradientBoosting training and feature importance
├── 03_inference_pipeline.py              # Production execution script
└── README.md                             # Project documentation
