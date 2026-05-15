# ActionSense 🦾
### Skeleton-Based Human Action Recognition using BiLSTM + Attention

A deep learning model that classifies human actions from 3D skeletal joint sequences — no raw video, no pixel-level noise. Just motion.

---

## Overview

ActionSense processes sequences of 3D body joint coordinates (25 joints × 3 axes × 50 frames) and predicts the action being performed. By working on skeleton data instead of raw video, the model is robust to lighting changes, backgrounds, and viewpoint variation.

**Final Accuracy: 90.23%** over 35 training epochs on the NTU RGB+D dataset.

---

## Architecture

```
Input (50 frames × 75 features)
        ↓
Feature Normalization
        ↓
Stacked BiLSTM (3 layers, 128 hidden units/direction)
  + Residual Connections + Layer Normalization
        ↓
Self-Attention (temporal)
        ↓
Dense → Softmax (action classes)
```

- **BiLSTM** — processes sequences forward and backward for richer temporal context  
- **Self-Attention** — learns to focus on the most informative frames  
- **Residual + LayerNorm** — stable training across deep layers  

---

## Dataset

[NTU RGB+D](https://arxiv.org/abs/1604.02808) — 56,880 skeleton action samples across 60 action classes.

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Sequence Length | 50 frames |
| Batch Size | 32 |
| Epochs | 35 (early stopping, patience 10) |
| Optimizer | Adam + OneCycleLR |
| Dropout | 0.3 |
| Loss | Cross-Entropy + Label Smoothing (0.1) |

---

## Results

| Epoch | Loss | Accuracy |
|---|---|---|
| 1 | 720.00 | 7.29% |
| 10 | 349.55 | 62.40% |
| 20 | 243.26 | 79.49% |
| 35 | 171.36 | **90.23%** |

---

## Stack

`Python` `PyTorch` `NumPy` `Matplotlib`

---
