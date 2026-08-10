# SkinSight

A skin lesion classification app built on a fine-tuned ResNet50 model, trained on the HAM10000 dermoscopy dataset.

## Overview

SkinSight classifies dermoscopic images into 7 lesion categories using transfer learning on ResNet50. The project compares three optimizers (Adam, AdamW, and SGD with Nesterov momentum) as its core research contribution — **SGD with Nesterov momentum achieved the best result at ~81% validation accuracy**, outperforming Adam/AdamW (~75%).

## Dataset

- **HAM10000** ("Human Against Machine with 10000 training images")
- 10,015 dermoscopy images across 7 lesion classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`

## Model Architecture

```
ResNet50 (ImageNet weights, no top)
  → GlobalAveragePooling2D
  → Dropout(0.5)
  → Dense(256, activation='relu')
  → Dropout(0.3)
  → Dense(7, activation='softmax')
```

- Input size: 224x224x3
- Trained with `preprocess_input` (ResNet50 ImageNet normalization) — not simple `1/255` rescaling
- Progressive layer unfreezing across 5 training phases
- Class weights capped to handle HAM10000's class imbalance (the `nv` class alone makes up ~67% of the dataset)

## Results

| Optimizer | Validation Accuracy |
|---|---|
| Adam | ~75% |
| AdamW | ~75% |
| **SGD + Nesterov** | **~81%** |

## Repository Contents

- `skinsight_training_reconstructed.ipynb` — training pipeline (data loading, augmentation, progressive unfreezing, all 5 training phases)
- `81__accuracy_sgd_phase_5.keras` — final trained model (SGD run, Phase 5 checkpoint)

## Requirements

```
tensorflow
pandas
numpy
scikit-learn
matplotlib
```

## Usage

1. Download the HAM10000 dataset (available on Kaggle).
2. Update `DATA_DIR` in the notebook to point to your dataset location.
3. Run the notebook cells in order — the pipeline auto-detects Kaggle or Colab environments.

To run inference with the pretrained model:

```python
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

model = tf.keras.models.load_model('81__accuracy_sgd_phase_5.keras')
# preprocess your 224x224 image with preprocess_input before predicting
```

## Disclaimer

This project is an academic research/final year project and is **not intended for clinical or diagnostic use**. Predictions should not be treated as medical advice.

## License

MIT
