import streamlit as st
import pandas as pd
import joblib
import os

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    layout="wide"
)

# -------------------- MINIMAL CLEAN CSS --------------------
st.markdown("""
<style>
.section {
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    background-color: #1c1f26;
}
.stButton>button {
    height: 45px;
    width: 100%;
    border-radius: 8px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "knn_heart_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "heart_scaler.pkl"))
expected_columns = joblib.load(os.path.join(BASE_DIR, "heart_columns.pkl"))

# -------------------- HEADER --------------------
st.title("Heart Disease Risk Assessment")
st.write("This system evaluates cardiovascular risk using clinical parameters and machine learning.")
st.write("Built By Shubham")

# -------------------- SIDEBAR --------------------
st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 18, 100, 40)
sex = st.sidebar.selectbox("Sex", ["M", "F"])
chest_pain = st.sidebar.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.sidebar.number_input("Cholesterol (mg/dL)", 100, 600, 200)

# ✅ FIXED fasting blood sugar
fasting_bs_label = st.sidebar.selectbox(
    "Fasting Blood Sugar",
    ["Normal (<=120 mg/dL)", "High (>120 mg/dL)"]
)
fasting_bs = 0 if "Normal" in fasting_bs_label else 1

resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.sidebar.slider("Maximum Heart Rate", 60, 220, 150)
exercise_angina = st.sidebar.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

# -------------------- MAIN LAYOUT --------------------
col1, col2 = st.columns([2, 1])

# -------------------- PATIENT SUMMARY --------------------
with col1:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("Patient Summary")

    st.write(f"Age: {age}")
    st.write(f"Sex: {sex}")
    st.write(f"Chest Pain Type: {chest_pain}")
    st.write(f"Resting Blood Pressure: {resting_bp} mm Hg")
    st.write(f"Cholesterol: {cholesterol} mg/dL")
    st.write(f"Fasting Blood Sugar: {fasting_bs_label}")
    st.write(f"Maximum Heart Rate: {max_hr}")
    st.write(f"Exercise-Induced Angina: {exercise_angina}")
    st.write(f"ST Depression: {oldpeak}")
    st.write(f"ST Slope: {st_slope}")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- PREDICTION --------------------
with col2:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("Risk Evaluation")

    if st.button("Run Assessment"):

        with st.spinner("Processing patient data..."):

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

            # Fill missing columns
            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0

            # Reorder columns
            input_df = input_df[expected_columns]

            # Scale
            scaled_input = scaler.transform(input_df)

            # Predict
            prediction = model.predict(scaled_input)[0]

        # -------------------- RESULT --------------------
        if prediction == 1:
            st.error("Elevated cardiovascular risk detected. A medical consultation is advised.")
        else:
            st.success("No significant cardiovascular risk detected based on current inputs.")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown("---")
st.write("This application provides a preliminary assessment and is not a substitute for professional medical advice.")
