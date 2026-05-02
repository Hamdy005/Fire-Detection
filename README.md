<div align="center">
  <h1>🔥 Fire, Smoke, and Non-Fire Detection</h1>
  <p><strong>A robust Convolutional Neural Network (CNN) for multi-class image classification.</strong></p>
</div>

---

## 📖 Project Overview

This project implements a multi-class image classification model capable of distinguishing between **Fire**, **Smoke**, and **Non-Fire** scenarios. The model was trained on a balanced dataset to ensure high accuracy and reduce class bias, resulting in a reliable detection system.

---

## 📊 Dataset & Visualizations

To build a robust model, we started with a high-quality dataset. We performed extensive exploratory data analysis (EDA) to validate the dataset's integrity before training.

### Data Inspection
Random sample grids were generated to visually inspect the images. This confirmed that the features for fire and smoke were visually distinct and well-represented.

<p align="center"> 
  <img src="https://github.com/user-attachments/assets/d885a4c7-9ee9-4a42-b7d5-d1b401ddfd45" alt="Random Samples" width="100%" />
</p>

### Class Distribution
We plotted bar and pie charts to confirm that the dataset is balanced across all three classes. A balanced dataset is crucial as it prevents the model from developing a bias toward the majority class.

<p align="center">
  <img src="https://github.com/user-attachments/assets/680572d8-85a3-4ab9-93b7-137b748b8e03" alt="Class Distribution" width="60%" />
</p>

---

## 🧠 Model Architecture & Training

The core of this project is a meticulously structured CNN designed to extract deep and meaningful features from images. 

### Architecture Highlights
- **Multiple Conv2D Blocks:** The network stacks several `Conv2D` layers (with 32, 64, 128, and 256 filters) coupled with `MaxPooling2D` and `BatchNormalization`. This deep, structured hierarchy effectively captures both low-level edges (like smoke textures) and high-level patterns (like fire shapes).
- **Fully Connected Layers:** The extracted features are flattened and passed through dense layers to make the final softmax classification.

### Solving Overfitting
Deep networks easily overfit. We tackled this by heavily regularizing the model:
- **Dropout Layers:** We applied progressive Dropout (ranging from 0.25 to 0.4) after pooling and dense layers to force the network to learn robust features and ignore noise.
- **L2 Regularization:** Used `L2(1e-5)` on the dense layers to penalize large weights and prevent over-reliance on specific nodes.

### Advanced Callbacks
Training was highly optimized using Keras callbacks:
- **`ReduceLROnPlateau`:** Automatically reduced the learning rate when validation loss stagnated, allowing the optimizer to make finer adjustments.
- **`EarlyStopping`:** Halted training when validation loss stopped improving, restoring the best weights to guarantee optimal generalization.

### Training Accuracy
<p align="center">
  <img src="https://github.com/user-attachments/assets/d94453cc-b033-402f-b648-ea02c6d7834a" alt="Training Accuracy" width="80%" />
</p>

---

## 🏆 Results

The model's extensive regularization and deep architecture paid off, resulting in outstanding performance on the unseen test dataset:

- **Test Accuracy:** `99.0%`
- **F1-Score:** `99.0%`

The model showcases high precision and recall, meaning it rarely misses a fire (false negatives) and rarely triggers false alarms (false positives).

---

## 🌐 Streamlit Web App Interface 

For a quick, interactive experience, we have developed a Streamlit application. It features a beautiful UI to upload images and see the model's predictions in real time.

Below are some screenshots of the Streamlit application and its prediction results:

#### App Homepage
<p align="center">
  <img src="https://github.com/user-attachments/assets/c93ab4e0-c72a-431d-8bee-1cd6d45951e3" alt="Home Screen" width="80%" />
</p>

#### Safe (Non-Fire) Detection
<p align="center"> 
  <img src="https://github.com/user-attachments/assets/b0f615a7-8377-4832-8ae1-cc912448c436" alt="Safe Detection" width="80%" />
</p>

#### Smoke Detection
<p align="center"> 
  <img src="https://github.com/user-attachments/assets/f0d442b0-8cf2-45f9-a5fa-043e601270e1" alt="Smoke Detection" width="80%" />
</p>

#### Fire Detection
<p align="center">
  <img src="https://github.com/user-attachments/assets/3d1b22c1-679b-4ce1-b46f-f110479a6969" alt="Fire Detection" width="80%" />
</p>

---

## 🚀 Installation & Usage

You can run this project locally or via the hosted version.

### 💻 Local Setup

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Run the Streamlit App:**
```bash
streamlit run app/app.py
```

### ☁️ Hosted Version
👉 [Try the Live App Here (https://dummy-streamlit-app-link.com)](#)
