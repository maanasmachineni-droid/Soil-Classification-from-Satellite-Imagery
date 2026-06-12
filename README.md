# 🌱 AI-Based Soil and Land Type Classification Using Deep Learning and Computer Vision

## 📌 Project Overview

This project is a Deep Learning-based Soil and Land Type Classification System that automatically identifies soil types from uploaded images. The system uses a Convolutional Neural Network (CNN) trained on soil image datasets and provides predictions through an interactive Streamlit web application.

The application allows users to upload soil images and instantly receive the predicted soil category along with confidence scores and probability distributions.

---

## 🎯 Objectives

* Automate soil type identification using Artificial Intelligence.
* Reduce the need for manual soil classification.
* Assist farmers, researchers, and agricultural organizations in soil analysis.
* Demonstrate the application of Deep Learning and Computer Vision in agriculture.

---

## 🚀 Features

* Upload soil images in JPG, JPEG, or PNG format.
* Real-time soil classification.
* Displays prediction confidence percentage.
* Shows probability distribution for all soil classes.
* User-friendly web interface built with Streamlit.
* Supports RGB, RGBA, and grayscale images.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Deep Learning Framework

* TensorFlow
* Keras

### Computer Vision

* OpenCV

### Data Processing

* NumPy
* Pandas

### Web Application

* Streamlit

### Image Processing

* Pillow (PIL)

---

## 📂 Project Structure

```text
Soil-Classification/
│
├── app.py
├── train_model.py
├── requirements.txt
│
├── models/
│   └── soil_classifier.h5
│
├── dataset/
│   ├── Arid_Soil/
│   ├── Black_Soil/
│   ├── Laterite_Soil/
│   ├── Mountain_Soil/
│   ├── Red_Soil/
│   └── Yellow_Soil/
│
└── README.md
```

---

## 🧠 Model Architecture

The classification model is based on a Convolutional Neural Network (CNN) consisting of:

* Convolution Layers
* ReLU Activation Functions
* Max Pooling Layers
* Dropout Layers
* Fully Connected Dense Layers
* Softmax Output Layer

The model learns visual patterns from soil images and predicts the most probable soil category.

---

## 📊 Soil Classes

The model is trained to classify the following soil types:

1. Arid Soil
2. Black Soil
3. Laterite Soil
4. Mountain Soil
5. Red Soil
6. Yellow Soil

---

## 🔄 Workflow

1. Collect soil image dataset.
2. Preprocess images.
3. Apply image augmentation.
4. Train CNN model.
5. Save trained model.
6. Upload soil image through Streamlit app.
7. Perform prediction.
8. Display soil type and confidence score.

---

## ▶️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/soil-classification.git

cd soil-classification
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📸 Sample Usage

1. Launch the Streamlit application.
2. Upload a soil image.
3. Click the **Predict Soil Type** button.
4. View:

   * Predicted Soil Type
   * Confidence Percentage
   * Probability Distribution Chart

---

## 🌾 Applications

* Smart Agriculture
* Precision Farming
* Soil Monitoring
* Land Resource Management
* Agricultural Research
* Environmental Studies

---

## ✅ Advantages

* Fast and automated soil classification.
* Reduces manual effort.
* Easy-to-use interface.
* High prediction accuracy.
* Scalable for larger datasets.
* Can be integrated into agricultural decision-support systems.

---

## 🔮 Future Enhancements

* Mobile Application Integration.
* GPS-based Soil Mapping.
* Real-time Camera Prediction.
* Fertilizer Recommendation System.
* Crop Recommendation Module.
* Cloud Deployment for Public Access.

---

## 📈 Results

The trained CNN model successfully classifies soil images and provides confidence scores for each prediction. The Streamlit application enables real-time interaction and visualization of prediction results.

---

## 👨‍💻 Author

Developed as part of a Deep Learning and Computer Vision project focused on intelligent soil and land type classification.

---

## 📜 License

This project is intended for educational and research purposes.
