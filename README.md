# 🔬 SkinSight — Skin Lesion Classification with Fine-Tuned ResNet50

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io)
[![Dataset](https://img.shields.io/badge/Dataset-HAM10000-blue?style=flat-square)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

**SkinSight** is a deep learning computer vision model for automated dermatological skin lesion classification. Built on a fine-tuned **ResNet50** architecture trained on the **HAM10000** dermoscopy dataset (10,015 images across 7 diagnostic categories), the project implements a 5-phase progressive unfreezing strategy and demonstrates how **SGD with Nesterov momentum** outperforms adaptive optimizers (Adam, AdamW) to achieve **~81% validation accuracy**.

---

## 📌 Architecture & Progressive Unfreezing

SkinSight employs transfer learning on a pre-trained ImageNet ResNet50 backbone with a custom classification head:

```text
Input (224x224x3, ResNet50 preprocess_input)
  │
  ▼
ResNet50 Backbone (ImageNet weights, no top)
  │
  ▼
GlobalAveragePooling2D
  │
  ▼
Dropout (rate = 0.5)
  │
  ▼
Dense (256 units, ReLU)
  │
  ▼
Dropout (rate = 0.3)
  │
  ▼
Dense (7 units, Softmax) ──> Diagnostic Lesion Probabilities
```

### 5-Phase Progressive Unfreezing Strategy

To prevent catastrophic forgetting of pre-trained feature extractors while adapting to dermoscopy textures:

1. **Phase 1 (Frozen Backbone):** Train classification head only with Adam ($10^{-3}$).
2. **Phase 2 (Stage 5 Unfrozen):** Unfreeze top residual stage (`conv5_block`) with Adam ($10^{-4}$).
3. **Phase 3 (Stages 4 & 5 Unfrozen):** Unfreeze intermediate stages (`conv4_block` + `conv5_block`) with Adam ($5 \times 10^{-5}$).
4. **Phase 4 (Optimizer Switch):** Unfreeze stage 3; switch from Adam to **SGD + Nesterov momentum** ($\mu = 0.9, \text{lr} = 5 \times 10^{-4}$) to escape sharp local minima.
5. **Phase 5 (Full Backbone Fine-Tuning):** Full network end-to-end fine-tuning with low-learning-rate SGD + Nesterov ($\text{lr} = 1.5 \times 10^{-4}$).

---

## 📊 Experimental Results

Across identical splits and class-weight balancing schedules, SGD with Nesterov momentum delivered superior generalization performance:

| Optimizer Strategy | Validation Accuracy | Generalization Notes |
| :--- | :---: | :--- |
| Adam | ~75% | Faster initial convergence; early plateau |
| AdamW | ~75% | Decoupled weight decay showed similar plateau |
| **SGD + Nesterov Momentum** | **~81%** | **Best convergence & flatter minima on unseen validation data** |

---

## 🩺 Dataset: HAM10000

The dataset contains 10,015 dermatoscopic images labeled across 7 categories:

| Class Code | Diagnostic Category | Notes |
| :--- | :--- | :--- |
| `nv` | Melanocytic nevi | ~67% of dataset (class imbalance handled via capped loss weighting) |
| `mel` | Melanoma | High-risk malignancy |
| `bkl` | Benign keratosis-like lesions | Solar lentigines / seborrheic keratoses |
| `bcc` | Basal cell carcinoma | Non-melanoma skin cancer |
| `akiec` | Actinic keratoses / Bowen's disease | Pre-cancerous squamous cell carcinoma |
| `vasc` | Vascular lesions | Angiomas, pyogenic granulomas |
| `df` | Dermatofibroma | Benign dermal lesions |

---

## 📁 Repository Structure

```text
SkinSight/
├── skinsight_training.ipynb     # End-to-end training pipeline, data loading, augmentation & 5 phases
├── SGD Momentum.keras           # Final trained model checkpoint (~81% validation accuracy)
├── .gitattributes               # Git LFS tracking rules for .keras model artifacts
├── LICENSE                      # MIT License
└── README.md                    # Project documentation
```

---

## 🚀 Quick Start & Inference

### 1. Requirements

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```

### 2. Loading the Model for Inference

```python
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

# Load trained checkpoint
model = tf.keras.models.load_model('SGD Momentum.keras')

# Preprocess image (224x224 RGB using ResNet50 ImageNet normalization)
# img_array = ... (shape: 1, 224, 224, 3)
img_preprocessed = preprocess_input(img_array)

# Predict probabilities across 7 classes
predictions = model.predict(img_preprocessed)
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
predicted_class = classes[np.argmax(predictions[0])]
print(f"Predicted class: {predicted_class} ({np.max(predictions[0]) * 100:.2f}%)")
```

---

## ⚠️ Academic Disclaimer

This project was developed for academic research and portfolio demonstration. It is **not** certified as a clinical medical device and is **not intended for real-world diagnostic use**.

---

## 📄 License

Distributed under the [MIT License](LICENSE).
