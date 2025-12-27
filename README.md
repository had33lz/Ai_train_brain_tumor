# Brain Tumor Detection using Deep Learning

This repository contains the implementation of a deep learning–based system for **binary classification of brain MRI images** into **tumor** and **no tumor** categories.  
The project was developed as part of the *Artificial Intelligence and Deep Learning* coursework and demonstrates both a **custom Convolutional Neural Network (CNN)** and a **transfer learning approach using ResNet50**.

---

## 📌 Problem Overview
Brain tumor detection from MRI scans is a challenging medical imaging task due to variations in tumor shape, size, and intensity. This project investigates whether transfer learning can outperform custom convolutional neural networks (CNNs) when training data is limited.

Four models were implemented and evaluated:

Model 1: Baseline Custom CNN

Model 2: Custom CNN with Batch Normalization and Dropout

Model 3: ResNet50 Transfer Learning (Frozen Backbone)

Model 4: ResNet50 Transfer Learning with Fine-Tuning (Best Model)

---

## 📂 Dataset
📊 Dataset

Source: Kaggle – Brain Tumor Dataset

Link: https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset

Data Type: Grayscale MRI images

Classes:

tumor

no_tumor

Dataset Split
Set	No Tumor	Tumor	Total
Training	1587	2013	3600
Testing	500	500	1000
The dataset is organized as and manualy splitted into test/train :

Brain_Tumor_Data_Set/ 
├── train/   
│ ├── no_tumor  
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

🏆 Results Summary
Model	Architecture	Accuracy
Model 1	Basic CNN	~50%
Model 2	CNN + BN + Dropout	~50%
Model 3	ResNet50 (Frozen)	~48%
Model 4	ResNet50 Fine-Tuned	~94%

The fine-tuned ResNet50 model achieved the best performance with balanced precision and recall across both classes.
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

1️⃣ Clone the repository
git clone https://github.com/your-username/brain_tumor_train.git
cd brain_tumor_train

2️⃣ Set up the environment   
python3 -m venv tfenv    
source tfenv/bin/activate    
pip install -r requirements.txt    

3️⃣ Train and evaluate    
python train.py    

1. Activate the virtual environment:   
bash   
source tfenv/bin/activate   
python train.py   


The dataset is organized as and manualy splitted into test/train :

Brain_Tumor_Data_Set/ 
├── train/   
│ ├── no_tumor  
│ └── tumor/   
└── test/   
├── no_tumor/  
└── tumor/  




👩‍💻 Author

Developed by Hadil
Artificial Intelligence & Deep Learning Coursework Project



⚠️ Notes

Dataset is not included due to size limitations.

This repository is intended for academic and research purposes only.
