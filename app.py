import streamlit as st
import numpy as np
import pickle

# ---------------- Page Settings ----------------
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="centered"
)

# ---------------- Title ----------------
st.title("💳 Credit Risk Prediction App")
st.write("Predict whether a loan application will be Approved or Rejected")

st.divider()

# ---------------- Load Model ----------------
try:
    model = pickle.load(open("credit_model.pkl","rb"))
except:
    model = None

# ---------------- Input Section ----------------
st.subheader("Applicant Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender",["Male","Female"])
    married = st.selectbox("Married",["Yes","No"])
    dependents = st.selectbox("Dependents",[0,1,2,3])
    education = st.selectbox("Education",["Graduate","Not Graduate"])
    self_employed = st.selectbox("Self Employed",["Yes","No"])
    property_area = st.selectbox("Property Area",["Urban","Semiurban","Rural"])

with col2:
    applicant_income = st.number_input("Applicant Income",0,100000,5000)
    coapplicant_income = st.number_input("Coapplicant Income",0,50000,0)
    loan_amount = st.number_input("Loan Amount",0,500000,150)
    loan_term = st.number_input("Loan Amount Term",0,500,360)
    credit_history = st.selectbox("Credit History",[1,0])

st.divider()

# ---------------- Prediction Button ----------------
predict = st.button("Predict Loan Status")

# ---------------- Prediction ----------------
if predict:

    if model is None:
        st.error("Model file not found. Add credit_model.pkl")
    else:

        # Convert categorical values
        gender = 1 if gender=="Male" else 0
        married = 1 if married=="Yes" else 0
        education = 1 if education=="Graduate" else 0
        self_employed = 1 if self_employed=="Yes" else 0

        property_map = {"Urban":2,"Semiurban":1,"Rural":0}
        property_area = property_map[property_area]

        # Model Input (11 Features)
        input_data = np.array([[
            gender,
            married,
            dependents,
            education,
            self_employed,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            credit_history,
            property_area
        ]])

        # Probability
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")

        st.write("Approval Probability:", round(probability*100,2),"%")
        st.progress(float(probability))

        # Rejection Conditions
        if credit_history == 0:
            st.error("❌ Loan Rejected (Poor Credit History)")

        elif applicant_income < 2500 and loan_amount > 300:
            st.error("❌ Loan Rejected (Low Income vs High Loan)")

        elif probability >= 0.6:
            st.success("✅ Loan Approved")

        else:
            st.error("❌ Loan Rejected")

st.divider()

st.caption("Machine Learning Credit Risk Prediction using Streamlit")