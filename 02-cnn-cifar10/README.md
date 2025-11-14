# CNN Image Classification (CIFAR-10)

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Python](https://img.shields.io/badge/Python-3.9-blue)

Achieved **82% accuracy** on CIFAR-10 using a **custom CNN** vs ANN baseline.

---

## Model Architecture

| Layer | Output Shape |
|------|--------------|
| Conv2D (32, 3x3) | 30x30x32 |
| MaxPool | 15x15x32 |
| Conv2D (64, 3x3) | 13x13x64 |
| MaxPool | 6x6x64 |
| Conv2D (64, 3x3) | 4x4x64 |
| Flatten + Dense | 10 classes |

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **82.1%** |
| Test Loss | 0.52 |
| Epochs | 10 |

![Training Curves](plots/training_curves.png)

---

## Run Locally

```bash
pip install -r requirements.txt
python cnn_cifar10.py
```

---

## Quick Test
```python
import tensorflow as tf
model = tf.keras.models.load_model("model/cifar10_cnn.h5")
# Predict on test[0]
```

---

## Tech Stack
- TensorFlow
- NumPy
- Matplotlib

---

