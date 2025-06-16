import streamlit as st
import pandas as pd
import numpy as np
import joblib


model = joblib.load("model.pkl")
st.title("Medical Insurance Charge Predictor")

st.markdown("Enter patient information to predict insurance cost:")

# User inputs
age = st.number_input("Age", min_value=18, max_value=100, step=1)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10, step=1)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Derived features (must match your training features!)
bmi_category = (
    'Underweight' if bmi < 18.5 else
    'Normal' if bmi < 25 else
    'Overweight' if bmi < 30 else
    'Obese'
)
age_group = (
    '18–25' if age <= 25 else
    '26–35' if age <= 35 else
    '36–50' if age <= 50 else
    '51–65'
)
age_smoker = age if smoker == 'yes' else 0
bmi_smoker = bmi if smoker == 'yes' else 0

# Prediction trigger
if st.button("Predict Insurance Charge"):
    input_df = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region,
        'bmi_category': bmi_category,
        'age_group': age_group,
        'age_smoker': age_smoker,
        'bmi_smoker': bmi_smoker
    }])

    log_pred = model.predict(input_df)
    final_pred = np.expm1(log_pred[0])

    st.success(f"Estimated Insurance Cost: ${final_pred:,.2f}")