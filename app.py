import streamlit as st
import pandas as pd
import joblib

df = pd.read_csv("fraud_cleaned.csv")
model = joblib.load("fraud_xgboost_pipeline.pkl")

categories = sorted(df["category"].unique())
cities = sorted(df["city"].unique())
states = sorted(df["state"].unique())
streets = sorted(df["street"].unique())

st.title("💳 Credit Card Fraud Detection")

st.write("Enter transaction details:")

amt = st.number_input("Transaction Amount")
city_pop = st.number_input("City Population")

lat = st.number_input("Latitude")
long = st.number_input("Longitude")
merch_lat = st.number_input("Merchant Latitude")
merch_long = st.number_input("Merchant Longitude")

hour = st.slider("Transaction Hour", 0, 23)
day = st.slider("Day of Month", 1, 31)
is_weekend = st.selectbox("Weekend?", [0,1])

category = st.selectbox("Category", categories)
city = st.selectbox("City", cities)
state = st.selectbox("State", states)
street = st.selectbox("Street", streets)

if st.button("Check Fraud"):

    input_data = pd.DataFrame({
        "amt":[amt],
        "city_pop":[city_pop],
        "lat":[lat],
        "long":[long],
        "merch_lat":[merch_lat],
        "merch_long":[merch_long],
        "hour":[hour],
        "day":[day],
        "is_weekend":[is_weekend],
        "category":[category],
        "city":[city],
        "state":[state],
        "street":[street]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    fraud_prob = probability
    safe_prob = 1 - probability

    if prediction == 1:
        st.error("⚠️ Fraud Detected!")
    else:
        st.success("✅ Transaction Safe")

    st.write(f"Fraud Probability: {fraud_prob:.2%}")
    st.write(f"Safe Probability: {safe_prob:.2%}")
