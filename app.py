import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("heart.csv")

X = df.drop('target', axis=1)
y = df['target']

# ---------------- TRAIN MODEL ----------------
model = DecisionTreeClassifier()
model.fit(X, y)

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>❤️ Heart Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter patient details to predict risk of heart disease</p>",
            unsafe_allow_html=True)

st.write("---")

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧍 Patient Information")

    age = st.slider("Age", 20, 80, 30)
    sex = st.selectbox("Sex", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 400, 200)

with col2:
    st.subheader("🫀 Medical Details")

    fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    thalach = st.slider("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

# ---------------- CONVERT INPUT ----------------
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# ---------------- CREATE INPUT DATA ----------------
input_data = pd.DataFrame([[
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal
]], columns=X.columns)

st.write("---")

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Risk"):

    # ensure correct column order
    input_data = input_data[X.columns]

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.write("---")

    if prediction[0] == 1:
        st.markdown(f"""
        <div style='background-color:#ff4b4b;padding:20px;border-radius:10px;text-align:center;'>
            <h2 style='color:white;'>⚠️ High Risk of Heart Disease</h2>
            <p style='color:white;'>Risk Probability: {probability:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background-color:#4CAF50;padding:20px;border-radius:10px;text-align:center;'>
            <h2 style='color:white;'>✅ Low Risk</h2>
            <p style='color:white;'>Risk Probability: {probability:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("About Project")
st.sidebar.info("This app predicts heart disease risk using a Machine Learning model (Decision Tree).")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>Made by Rajan</p>", unsafe_allow_html=True)