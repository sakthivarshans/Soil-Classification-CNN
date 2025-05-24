# Soil-Classification-CNN


🧪 Soil Type Classification using CNN

This project is built to classify images of **soil types** into four categories — **Alluvial Soil, Black Soil, Clay Soil, and Red Soil** — using a **Convolutional Neural Network (CNN)**. The model is trained on image data and labels provided in CSV format and implemented in a **Kaggle Notebook**.

---

 🧠 Why This Project?

Soil classification is critical in agriculture, irrigation planning, and crop suitability assessment. Using AI-based image classification models:

- We reduce human error in soil identification.
- Automate soil-type detection with just an image.
- Enable farmers and agronomists to use mobile or drone images for quick assessment.

---

 🗃️ Dataset Overview

The dataset is hosted on Kaggle and includes:

| Component | Path |
|----------|------|
| 📁 Train Images | `/kaggle/input/soil-classification/soil_classification-2025/train` |
| 📁 Test Images | `/kaggle/input/soil-classification/soil_classification-2025/test` |
| 📄 Train Labels | `/kaggle/input/soil-classification/soil_classification-2025/train_labels.csv` |
| 📄 Test IDs | `/kaggle/input/soil-classification/soil_classification-2025/test_ids.csv` |
| 📄 Sample Submission | `/kaggle/input/soil-classification/soil_classification-2025/sample_submission.csv` |

---

 🧑‍💻 Technologies Used

| Tool | Purpose |
|------|---------|
| **Python** | Main programming language |
| **TensorFlow / Keras** | Building and training the CNN model |
| **Pandas & NumPy** | Data handling and manipulation |
| **Matplotlib** | Visualizing training performance |
| **Kaggle Notebook** | Cloud-based coding environment with GPU support |

---

 🔍 Project Workflow

 🔹 1. Load Dataset

- CSV files (train labels, test IDs, sample submission) are read using `pandas`.
- Images from train and test folders are loaded and resized to `(128, 128)`.

🔹 2. Preprocess Data

- Normalized pixel values (0–1 range).
- Labels are mapped to integer classes for training.

🔹 3. Build CNN Model

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 soil classes
])

