import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Boston House Price Prediction")

st.title("🏠 Boston House Price Prediction")

st.write("Enter the house details below to predict the house price.")

# User Inputs
CRIM = st.number_input("Crime Rate (CRIM)", value=0.10)
ZN = st.number_input("Residential Land Zoned (ZN)", value=18.0)
INDUS = st.number_input("Industrial Area (INDUS)", value=2.31)
CHAS = st.selectbox("Charles River (CHAS)", [0, 1])
NOX = st.number_input("Nitric Oxide Concentration (NOX)", value=0.538)
RM = st.number_input("Average Number of Rooms (RM)", value=6.57)
AGE = st.number_input("Age of House (AGE)", value=65.2)
DIS = st.number_input("Distance to Employment Centres (DIS)", value=4.09)
RAD = st.number_input("Accessibility to Highways (RAD)", value=1)
TAX = st.number_input("Property Tax Rate (TAX)", value=296)
PTRATIO = st.number_input("Pupil-Teacher Ratio (PTRATIO)", value=15.3)
B = st.number_input("Proportion of Blacks (B)", value=396.90)
LSTAT = st.number_input("Lower Status Population (%)", value=4.98)


prediction = None


if st.button("Predict House Price"):

    # Apply the same preprocessing used during training
    CRIM = np.log1p(CRIM)
    ZN = np.log1p(ZN)
    DIS = np.log1p(DIS)
    B = np.log1p(B)
    LSTAT = np.log1p(LSTAT)

    sample = np.array([[CRIM, ZN, INDUS, CHAS, NOX,
                        RM, AGE, DIS, RAD, TAX,
                        PTRATIO, B, LSTAT]])

    sample = scaler.transform(sample)

    prediction = model.predict(sample)




if prediction is not None:
    st.metric(
        label="Predicted House Price",
        value=f"${prediction[0] * 1000:,.2f}"
    )

    st.caption(f"MEDV = {prediction[0]:.2f}")

