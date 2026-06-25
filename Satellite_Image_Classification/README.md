# 🛰️ Satellite Image Land-Use Classification & Temporal Change Detection

## 📌 Overview

This project implements a deep learning framework for **satellite image land-use classification** and **temporal change detection** using the EuroSAT dataset. The system combines transfer learning, embedding-based similarity analysis, explainable AI techniques, and an interactive Streamlit dashboard for real-time inference.

The project was developed as part of the **Celebal Technologies Data Science Internship Program**.

---

## 🚀 Features

### Land-Use Classification

* Baseline 3-layer CNN implementation
* Transfer Learning using ResNet18
* Two-phase fine-tuning strategy
* Multi-class classification across 10 EuroSAT categories

### Temporal Change Detection

* Feature embedding extraction using ResNet18 backbone
* Cosine similarity-based change detection
* ROC-based threshold selection
* Visual change heatmap generation

### Interactive Dashboard

* Streamlit-based user interface
* Upload before/after satellite images
* Real-time land-use prediction
* Similarity score computation
* Change/No-Change decision
* Heatmap visualization
* Multiple threshold operating modes

### Explainable AI & Advanced Analysis

* GradCAM visualizations
* t-SNE embedding visualization
* Class imbalance experiments
* Error analysis on misclassified samples

---

## 📂 Dataset

### EuroSAT RGB Dataset

* 27,000 satellite images
* 10 land-use classes
* Sentinel-2 imagery
* Used for training, validation, and testing

Classes:

* AnnualCrop
* Forest
* HerbaceousVegetation
* Highway
* Industrial
* Pasture
* PermanentCrop
* Residential
* River
* SeaLake

### UC Merced Land Use Dataset

* 2,100 aerial images
* 21 land-use categories
* Used for generalization evaluation

---

## 🏗️ Project Workflow

```text
Satellite Images
        │
        ▼
   Preprocessing
        │
        ▼
   Baseline CNN
        │
        ▼
 Transfer Learning
    (ResNet18)
        │
        ▼
 Land-Use Classification
        │
        ▼
 Feature Extraction
        │
        ▼
 Cosine Similarity
        │
        ▼
 Change Detection
        │
        ▼
 Heatmap Generation
        │
        ▼
 Streamlit Dashboard
```

---

## 🧠 Model Architecture

### Baseline CNN

* 3 Convolutional Layers
* ReLU Activation
* Max Pooling
* Fully Connected Classification Layer

### Transfer Learning Model

* ResNet18 pretrained on ImageNet

* Phase 1:

  * Frozen Backbone
  * Train Classification Head
  * 3 Epochs

* Phase 2:

  * Unfreeze Layer4
  * Lower Learning Rate
  * 5 Epochs

---

## 📊 Results

### Classification Performance

| Model              | Validation Accuracy |
| ------------------ | ------------------- |
| Baseline CNN       | 79.14%              |
| ResNet18 (Phase 1) | 90.86%              |
| ResNet18 (Phase 2) | 97.06%              |

### Final Metrics

| Metric               | Score  |
| -------------------- | ------ |
| Validation Accuracy  | 97.06% |
| Macro F1 Score       | 0.971  |
| Change Detection AUC | 0.9769 |

---

## 🔍 Change Detection Pipeline

1. Remove classification head from ResNet18
2. Extract 512-dimensional embeddings
3. Compute cosine similarity between image pairs
4. Select threshold using ROC analysis
5. Generate change/no-change decision
6. Produce visual heatmaps

---

## 🎯 Bonus Implementations

### Bonus A — GradCAM Visualization

Implemented GradCAM to visualize image regions contributing to classification decisions.

### Bonus B — Multi-Threshold Toggle

Added:

* High Recall Mode
* Balanced Mode
* High Precision Mode

for flexible change detection sensitivity.

### Bonus C — Embedding Visualization

Projected ResNet18 embeddings into 2D using t-SNE to visualize class separability.

### Bonus D — Imbalance Experiment

Performed class imbalance simulation and applied weighted-loss mitigation.

---

## 🖥️ Streamlit Dashboard

Dashboard Features:

* Satellite image upload
* Land-use classification
* Confidence scores
* Similarity computation
* Change detection
* Heatmap visualization
* Threshold selection

Run locally without internet connectivity after setup.

---

## 📁 Project Structure

```text
SATELLITE_IMAGE_CLASSIFICATION
│
├── models
│   └── best_resnet18.pt
│
├── notebooks
│   └── satellite_image_classification.ipynb
│
├── outputs
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── gradcam_1.png
│   ├── gradcam_2.png
│   ├── gradcam_3.png
│   ├── tsne_resnet18.png
│   ├── tsne_classwise.png
│   ├── threshold_comparison.png
│   ├── threshold_shift.png
│   └── imbalance_experiment.png
│
├──report
│   ├── final report.pdf
├── app.py
├── utils.py
├── requirements.txt
└── README.md
```

## 📄 Project Report

The complete project report is available in the `report` folder.

## 🎥 Demo Video

The project demonstration video can be viewed here:

https://drive.google.com/file/d/1O5kFxJj0E0VJup14pgPKXZpMUWvBrqn6/view?usp=sharing

---

## ⚙️ Installation

### Clone Repository

```bash
git clone <repository-url>
cd SATELLITE_IMAGE_CLASSIFICATION
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Dashboard

```bash
streamlit run app.py
```

Open the generated local URL in your browser.

---

## 📈 Key Learnings

* Transfer Learning for Remote Sensing
* Satellite Image Classification
* Embedding-Based Change Detection
* Explainable AI using GradCAM
* Feature Space Visualization using t-SNE
* Streamlit Application Development
* Model Evaluation and Error Analysis

---

## 🔮 Future Work

* Geographic block-based train/test splitting
* Vision Transformer (ViT) based classification
* Semantic segmentation for precise change localization
* Multi-temporal satellite image analysis
* Deployment on cloud platforms

---


