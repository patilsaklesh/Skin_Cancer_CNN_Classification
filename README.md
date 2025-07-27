# 🧠 Skin Cancer Classification using CNN

## 📌 Project Overview

This project is an end-to-end deep learning solution to classify **skin lesions** as **benign or malignant** using **Convolutional Neural Networks (CNNs)**. It leverages data preprocessing, data augmentation, a custom CNN architecture, and evaluation metrics to predict skin cancer from image data.

---

## 🚀 Features

✅ Image preprocessing using Keras `ImageDataGenerator`  
✅ Data Augmentation for improved generalization  
✅ Custom CNN architecture built using TensorFlow/Keras  
✅ Binary classification: **Benign (0)** vs **Malignant (1)**  
✅ Model evaluation using Accuracy, Confusion Matrix, and Classification Report  
✅ Model export in `.h5` format for deployment  
✅ Ready for integration with **Streamlit** app

---

## 🗂️ Project Structure

```
Skin_Cancer_Classification_CNN/
│
├── dataset/ # Contains 'train/' and 'test/' folders
│ ├── train/
│ │ ├── benign/
│ │ └── malignant/
│ └── test/
│ ├── benign/
│ └── malignant/
│
├── notebook/
│ └── skin_cancer_classification_CNN.ipynb 
│
├── model/
│ └── skin_cancer_cnn.h5 
│
├── app.py (soon to be implemented)
└── README.md
```

## 🧠 CNN Architecture Summary

- 3 Convolutional + MaxPooling layers  
- `Flatten → Dense (512) → Dropout → Dense (1)`  
- Activation functions: **ReLU**, **Sigmoid**  
- Optimizer: **Adam**  
- Loss: **Binary Crossentropy**  
- Callbacks: **EarlyStopping**, **ReduceLROnPlateau**

---

## 📊 Model Evaluation

✅ Accuracy  
✅ Confusion Matrix  
✅ Precision, Recall, and F1-Score

## 💡 Future Improvements
- Use transfer learning (e.g., EfficientNet, ResNet)
- Incorporate Grad-CAM for model interpretability
- Deploy the model using Streamlit or Flask
- Host it on AWS EC2 or SageMaker
- Automate preprocessing with a pipeline

## 💾 Model Saving

Model is saved as:
```python
model.save("skin_cancer_cnn.h5")