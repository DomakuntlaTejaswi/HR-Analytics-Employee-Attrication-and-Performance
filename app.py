import streamlit as st
import joblib
import pandas as pd

# Load trained pipeline
model = joblib.load("attrition_pipeline.joblib")

st.title("💼 Employee Attrition Prediction (HR Analytics)")

# User input fields
Age = st.number_input("Age", 18, 60, 30)
MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000)
YearsAtCompany = st.number_input("Years at Company", 0, 40, 5)
JobRole = st.selectbox("Job Role", [
    "Sales Executive","Research Scientist","Laboratory Technician",
    "Manufacturing Director","Healthcare Representative","Manager",
    "Sales Representative","Research Director","Human Resources"
])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
OverTime = st.selectbox("OverTime", ["Yes", "No"])

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame({
        "Age":[Age],
        "MonthlyIncome":[MonthlyIncome],
        "YearsAtCompany":[YearsAtCompany],
        "JobRole":[JobRole],
        "MaritalStatus":[MaritalStatus],
        "OverTime":[OverTime]
    })

    prediction = model.predict(input_data)[0]
    result = "❌ Employee likely to leave" if prediction == 1 else "✅ Employee likely to stay"
    st.subheader(result)
