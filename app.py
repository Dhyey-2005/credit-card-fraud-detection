import streamlit as st
import pandas as pd
import joblib

model = joblib.load("fraud_xgboost_pipeline.pkl")


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

states = ['Achille', 'Acworth', 'Adams', 'Afton', 'Akron', 'Albany', 'Albuquerque', 'Alder', 'Aledo', 'Alexandria', 'Allenhurst', 'Allentown', 'Alpharetta', 'Altair', 'Alton', 'Altona', 'Altonah', 'Alva', 'Amanda', 'American Fork', 'Amorita', 'Amsterdam', 'Andrews', 'Angwin', 'Annapolis', 'Apison', 'Arcadia', 'Arlington', 'Armagh', 'Armonk', 'Arnold', 'Arvada', 'Ash Flat', 'Ashfield', 'Ashford', 'Ashland', 'Atglen', 'Athena', 'Atlantic', 'Auburn', 'Aurora', 'Avera', 'Avoca', 'Azusa', 'Bagley', 'Bailey', 'Ballwin', 'Barnard', 'Barneveld', 'Barnstable']
city = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
category = ['entertainment', 'food_dining', 'gas_transport', 'grocery_net', 'grocery_pos', 'health_fitness', 'home', 'kids_pets', 'misc_net', 'misc_pos', 'personal_care', 'shopping_net', 'shopping_pos', 'travel']
streets = ['000 Jennifer Mills', '0005 Morrison Land', '00315 Ashley Valleys', '00378 Sarah Burgs Suite 106', '0043 Henry Plaza', '005 Cody Estates', '0069 Robin Brooks Apt. 695', '00821 Joanna Meadow', '010 Salazar Walk', '010 Weaver Land', '0107 Clements Point', '0110 Ashley Forest', '01479 Murray Circle', '01505 Amy Stravenue', '0157 Samuel Mission Suite 379', '01770 Kevin Lodge Suite 190', '0182 Owens Burgs Suite 480', '0189 Emily Prairie', '01892 Patricia Vista Apt. 828', '019 Kimberly Light Apt. 039', '02018 Gary Key Apt. 911', '0207 Griffith Plains Apt. 544', '02110 Lucas Freeway Suite 517', '022 Moore Island', '024 Williams Parkway', '025 White Fork Apt. 633', '030 Seth Divide Suite 355', '03030 White Lakes', '03090 Fisher Forges Apt. 200', '033 Tara Brook Suite 523', '03368 Michelle Trail', '034 Kimberly Mountains', '03512 Jackson Ports', '0356 Sarah Light', '0362 Anderson Wall', '0371 Aimee Neck Suite 856', '0374 Courtney Islands Apt. 400', '03921 Cole Mission Suite 882', '04139 Johnson Prairie Suite 401', '0423 Kirby Field Suite 623', '043 Hanson Turnpike', '046 Michelle Fort Suite 314', '04611 Sandra Spring Suite 059', '0467 Jerry Pines Apt. 640', '047 Kevin Haven', '0495 Baker Manors', '04975 Allison Shoal', '05050 Rogers Well Apt. 439', '053 Kim Valley Suite 928', '0537 Margaret Common Suite 526']

category = st.selectbox("Category", category)
city = st.selectbox("City", city)
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
