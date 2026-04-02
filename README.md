# 🔬 Gastrointestinal Disease Classification using Deep Learning

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![accuracy](https://img.shields.io/badge/Test%20Accuracy-95%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

A deep learning model that classifies gastrointestinal diseases from endoscopy images with **95% test accuracy**, using transfer learning on EfficientNetV2S pretrained on ImageNet.

---

## 📌 Project Overview

Early and accurate detection of gastrointestinal conditions can significantly improve patient outcomes. This project builds an 8-class image classifier trained on real endoscopy images, capable of identifying conditions such as polyps, esophagitis, ulcerative colitis, and normal tissue types — automatically.

This was developed as my **Final Year Project** at the University of Haripur (AI Engineering).

---

## 🎯 Classes Detected

| Class | Description |
|-------|-------------|
| `ulcerative-colitis` | Inflammatory bowel disease affecting the colon |
| `polyps` | Abnormal tissue growths that may become cancerous |
| `esophagitis` | Inflammation of the esophagus lining |
| `dyed-resection-margins` | Post-surgical tissue markings |
| `normal-cecum` | Healthy cecum tissue |
| `normal-pylorus` | Healthy pylorus tissue |
| `normal-z-line` | Healthy gastroesophageal junction |

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **95%** |
| Training Images | 6,432 |
| Validation Images | 804 |
| Test Images | 804 |
| Total Dataset Size | 8,040 images |
| Image Size | 224 × 224 px |

**Confusion Matrix highlights:**
- Perfect classification on `normal-pylorus` and `normal-cecum`
- Minor confusion between `polyps` and `esophagitis` (expected given visual similarity)

---

## 🏗️ Model Architecture

```
Input (224×224×3)
     ↓
EfficientNetV2S (pretrained on ImageNet, fully fine-tuned)
     ↓
Global Average Pooling
     ↓
Flatten
     ↓
Dense(8, activation='softmax')
     ↓
Output: 8-class probabilities
```

**Why EfficientNetV2S?**
- State-of-the-art accuracy-to-parameter ratio
- Strong feature extraction from pretrained ImageNet weights
- Fine-tuned all layers for domain-specific medical image adaptation

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=0.001, weight_decay=0.004) |
| Loss | Categorical Crossentropy |
| Epochs | Up to 100 (early stopping) |
| Batch Size | 64 |
| Callbacks | ReduceLROnPlateau, ModelCheckpoint, EarlyStopping |

---

## 🛠️ Tech Stack

- **Python 3.11**
- **TensorFlow 2.18** / Keras
- **scikit-learn** — metrics, train/test split
- **Matplotlib & Seaborn** — visualization
- **NumPy / Pillow** — image processing
- **Google Colab** — GPU training (T4)

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/syedaatiqakanwal-sak/gastrointestinal-disease-classification.git
cd gastrointestinal-disease-classification
```

### 2. Install dependencies
```bash
pip install tensorflow scikit-learn matplotlib seaborn numpy pillow
```

### 3. Prepare your dataset
Download the [Kvasir Dataset](https://datasets.simula.no/kvasir/) and place it as:
```
dataset/
├── ulcerative-colitis/
├── polyps/
├── esophagitis/
├── dyed-resection-margins/
├── normal-cecum/
├── normal-pylorus/
└── normal-z-line/
```

### 4. Run the notebook
Open `colon_cancer_v2.ipynb` in Google Colab or Jupyter and run all cells.

---

## 📁 Project Structure

```
├── colon_cancer_v2.ipynb     # Main training notebook
├── README.md                 # Project documentation
└── training_weights/
    └── best/
        └── model_best_val_accuracy.weights.h5
```

---

## 👩‍💻 Author

**Syeda Atiqa Kanwal** — AI Engineer | University Gold Medallist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/syeda-atiqa-kanwal-838490390)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat-square&logo=github&logoColor=white)](https://github.com/syedaatiqakanwal-sak)

---

## 📄 License

MIT License — free to use with attribution.
