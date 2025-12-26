# Brain Tumor Detection using Deep Learning

This repository contains the implementation of a deep learning–based system for **binary classification of brain MRI images** into **tumor** and **no tumor** categories.  
The project was developed as part of the *Artificial Intelligence and Deep Learning* coursework and demonstrates both a **custom Convolutional Neural Network (CNN)** and a **transfer learning approach using ResNet50**.

---

## 📌 Problem Overview
Brain tumor detection from MRI scans is a critical task in medical image analysis. Manual diagnosis is time-consuming and depends heavily on expert interpretation. This project explores how deep learning models can assist in automatically identifying tumor presence from grayscale MRI images.

---

## 📂 Dataset
- **Type:** Brain MRI images (grayscale)
- **Task:** Binary classification (Tumor / No Tumor)

### Dataset Distribution
| Set | No Tumor | Tumor |
|----|---------|-------|
| Training | 1587 | 2013 |
| Testing | 500 | 500 |

The dataset is organized as:
Brain_Tumor_Data_Set/

├── train/

│ ├── no_tumor/

│ └── tumor/

└── test/

├── no_tumor/

└── tumor/


---

## ⚙️ Methods

### 1️⃣ Custom CNN
A Sequential CNN model was designed from scratch using:
- Convolutional layers with Batch Normalization
- MaxPooling and Dropout to reduce overfitting
- Global Average Pooling for feature reduction

This model learns domain-specific features directly from the MRI images.

### 2️⃣ Transfer Learning (ResNet50)
A pre-trained **ResNet50** model (ImageNet weights) was used:
- Backbone initially frozen
- Grayscale images converted to RGB inside the model
- Fine-tuning applied to higher layers with a low learning rate

This approach leverages pre-learned visual features to improve generalization.

---

## 🧪 Evaluation
Models are evaluated on the **test set** using:
- Accuracy
- Precision
- Recall
- AUC
- Confusion Matrix
- Classification Report

Class imbalance in the training data is handled using **class weighting**.

---

## 🚀 Training Environment
- **Framework:** TensorFlow / Keras
- **Execution:** Local training using **WSL + VS Code**
- **GPU:** NVIDIA RTX 3060
- **OS:** Ubuntu (WSL2)

This setup avoids cloud I/O bottlenecks and enables faster, stable training.

---

## 📁 Repository Structure


## ▶️ How to Run

1. Activate the virtual environment:
```bash
source tfenv/bin/activate
python train.py


👩‍💻 Author

Developed by Hadil
Artificial Intelligence & Deep Learning Coursework Project