import streamlit as st
import pandas as pd
import joblib
import os

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Heart Risk Assessment",
    layout="centered"
)

# -------------------- LOAD MODEL --------------------
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "knn_heart_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "heart_scaler.pkl"))
expected_columns = joblib.load(os.path.join(BASE_DIR, "heart_columns.pkl"))

# -------------------- HEADER --------------------
st.title("Heart Disease Risk Assessment")
st.write("Provide patient details to evaluate cardiovascular risk.")
st.write("Built with care by Shubham")

st.markdown("---")

# -------------------- FORM --------------------
with st.form("patient_form"):

    st.subheader("Basic Information")

    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])

    st.subheader("Clinical Parameters")

    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

    fasting_bs_label = st.selectbox(
        "Fasting Blood Sugar",
        ["Normal (<=120 mg/dL)", "High (>120 mg/dL)"]
    )
    fasting_bs = 0 if "Normal" in fasting_bs_label else 1

    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    submit = st.form_submit_button("Run Assessment")

# -------------------- PREDICTION --------------------
if submit:

    with st.spinner("Evaluating patient data..."):

        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        input_df = pd.DataFrame([raw_input])

        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[expected_columns]
        scaled_input = scaler.transform(input_df)

        prediction = model.predict(scaled_input)[0]

    st.markdown("---")

    if prediction == 1:
        st.error("Elevated cardiovascular risk detected. Medical consultation is recommended.")
    else:
        st.success("No significant cardiovascular risk detected based on provided inputs.")

# -------------------- FOOTER --------------------
st.markdown("---")
st.write("This tool is intended for preliminary assessment and does not replace professional medical advice.")
