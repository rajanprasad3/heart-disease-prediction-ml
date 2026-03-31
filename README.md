# ❤️ Heart Disease Prediction App

## 📌 Project Overview
This project aims to predict the risk of heart disease using patient medical data.  
A machine learning model is trained on a healthcare dataset to classify whether a person is at risk or not.

The project also includes a simple and interactive web application built using Streamlit for real-time predictions.
Live app - https://heart-disease-prediction-ml2.streamlit.app/
---

## 🎯 Objective
- Analyze healthcare data
- Build a machine learning model for prediction
- Create a user-friendly interface for real-time risk prediction

---

## 📊 Dataset
- Source: Kaggle (Heart Disease Dataset)
- The dataset contains medical attributes such as:
  - Age
  - Sex
  - Chest Pain Type
  - Blood Pressure
  - Cholesterol
  - Maximum Heart Rate
  - And other clinical features

---

## ⚙️ Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit

---

## 🧠 Machine Learning Model
- Model Used: Decision Tree Classifier
- Train-Test Split: 70% / 30%
- Achieved Accuracy: ~97%

---

## 📈 Key Features
- Data preprocessing and analysis
- Feature importance visualization
- Heart disease prediction model
- Interactive Streamlit web app
- Real-time prediction with probability score

---

## 🖥️ Streamlit App
The app allows users to:
- Input patient details
- Get instant prediction (High Risk / Low Risk)
- View probability of heart disease

---

## 🚀 How to Run the Project

pip install -r requirements.txt

streamlit run app.py

#📂 Project Structure 

heart-disease-prediction-app/
│
├── app.py
├── heart.csv
├── requirements.txt
├── heart_disease_prediction.ipynb
└── README.md
