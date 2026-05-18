# 🌱 Plant Disease Detection using Machine Learning & Deep Learning

A comparative study on plant disease detection using traditional Machine Learning and Deep Learning techniques on the PlantVillage dataset.

---

# 📌 Project Overview

Plant diseases significantly affect agricultural productivity and food quality worldwide. Early and accurate detection of plant diseases can help farmers reduce crop loss and improve sustainable farming practices.

This project presents a comparative analysis of multiple Machine Learning and Deep Learning models for automated plant disease detection using leaf images.

The study evaluates and compares the performance of:

- Convolutional Neural Network (CNN)
- ResNet50 (Transfer Learning)
- Random Forest
- AdaBoost
- K-Nearest Neighbors (KNN)

The models were trained and tested on the PlantVillage dataset containing healthy and diseased plant leaf images.

---

# 🧠 Research Paper

## Title
**A Comparative Study of Plant Disease Detection Using Machine Learning**

## Authors
- Khandoker Md. Ragib Ahsan
- Asif Al Mahfuz
- Anika Afrin

## Institution
Department of Computer Science and Engineering  
BRAC University

---

# 🎯 Objectives

- Detect plant diseases automatically using AI techniques
- Compare traditional ML algorithms with Deep Learning models
- Analyze performance using evaluation metrics
- Improve agricultural productivity through intelligent disease detection

---

# 📂 Dataset

## Dataset Used
**PlantVillage Dataset**

## Dataset Features

- 54,305 plant leaf images
- 38 disease classes
- 14 crop species
- Includes healthy and diseased leaves

## Image Categories

- Bacterial diseases
- Viral diseases
- Mold diseases
- Mite-related diseases
- Healthy leaves

---

# ⚙️ Data Preprocessing

The following preprocessing techniques were applied:

- Image resizing
- Normalization
- PCA dimensionality reduction
- Horizontal flipping
- Random cropping
- Gaussian noise augmentation
- Color channel flipping

## Dataset Split

| Dataset | Percentage |
|---|---|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

---

# 🤖 Models Used

## 1. Custom CNN

A deep learning CNN architecture with:

- Conv2D
- Batch Normalization
- MaxPooling
- Dropout
- GlobalAveragePooling

---

## 2. ResNet50 (Transfer Learning)

Pretrained ResNet50 model using ImageNet weights.

---

## 3. Random Forest

Traditional ensemble-based ML classifier using PCA-reduced features.

---

## 4. AdaBoost

Boosting-based classifier using Decision Tree stumps.

---

## 5. K-Nearest Neighbors (KNN)

Distance-based classification using Euclidean distance.

---

# 📊 Performance Comparison

| Model | Test Accuracy |
|---|---|
| ResNet50 | 98.22% |
| Custom CNN | 81.5% |
| KNN | 67.52% |
| Random Forest | 66.82% |
| AdaBoost | 24.11% |

---

# 📈 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

# 🏆 Best Performing Model

## ✅ ResNet50

Achieved the highest test accuracy:

```bash
98.22%
