import streamlit as st
import joblib
import pandas as pd

# Load model
model_data = joblib.load('student_model_package.pkl')

model = model_data['model']
scaler = model_data['scaler']
columns = model_data['columns']

st.title("🎓 Student Pass/Fail Prediction")
st.write("Enter student data:")

input_data = {}

for col in columns:
    input_data[col] = st.number_input(f"{col}", value=0.0)

if st.button("Predict"):
    df = pd.DataFrame([input_data])

    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    if prediction == 1:
        st.success(f"✅ Pass (Probability: {probability:.2f})")
    else:
        st.error(f"❌ Fail (Probability: {probability:.2f})")
