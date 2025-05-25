# 🧠 Convolutional Neural Network (CNN) from Scratch using TensorFlow 1.x on MNIST

This project implements a **Convolutional Neural Network (CNN)** from scratch using **low-level TensorFlow 1.x operations** to classify handwritten digits from the **MNIST dataset**.

---

## 📌 Highlights

- ✅ Implemented CNN using `tf.nn.conv2d`, `tf.nn.bias_add`, `tf.nn.relu`, `tf.nn.max_pool`, etc.
- ✅ Trained on MNIST dataset (28x28 grayscale digits)
- ✅ Custom architecture with two convolutional layers and fully connected layers
- ✅ Accuracy reported on both training and testing data
- ✅ No Keras used – built fully using raw TensorFlow 1.x

---

## 📊 Dataset

- **MNIST**: 70,000 handwritten digits (28x28 pixels)
  - Training: 60,000 images
  - Testing: 10,000 images
  - 10 classes (digits 0–9)

---

## 🧠 Model Architecture

Input: [None, 784] → Reshaped to [None, 28, 28, 1]

1️⃣ Conv Layer 1: 32 filters, 5x5, stride=1 → ReLU
2️⃣ Max Pooling 1: 2x2

3️⃣ Conv Layer 2: 64 filters, 5x5, stride=1 → ReLU
4️⃣ Max Pooling 2: 2x2

5️⃣ Flatten → Fully Connected: 1024 units → ReLU
6️⃣ Dropout (keep_prob = 0.6 during training)

7️⃣ Output: Dense(10) → Logits → Softmax for prediction