import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("risk_model.pkl", "rb"))

# Title
st.title("🛒 E-commerce Risk Predictor")

st.write("Enter order details below")

# Inputs
payment_value = st.number_input(
    "Payment Amount",
    min_value=100,
    max_value=10000,
    value=1000
)

review = st.slider(
    "Review Score",
    1,
    5,
    3
)

city = st.selectbox(
    "Customer City",
    ["Mumbai", "Delhi", "Bangalore"]
)

payment_type = st.selectbox(
    "Payment Type",
    ["credit_card", "debit_card", "upi"]
)

product_category = st.selectbox(
    "Product Category",
    ["electronics", "fashion", "home"]
)

month = st.selectbox(
    "Month",
    ["January","February","March","April","May","June",
     "July","August","September","October","November","December"]
)

# Predict button
if st.button("Predict Risk"):

    # Create dataframe
    new_order = pd.DataFrame({
        "payment value":[payment_value],
        "review":[review],
        "customer_city":[city],
        "payment type":[payment_type],
        "product category":[product_category],
        "month":[month]
    })

    # Convert categorical data
    new_order = pd.get_dummies(new_order)

    # Match training columns
    new_order = new_order.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(new_order)

    # Show result
    if prediction[0] == 1:
        st.error("⚠️ Risky Order")
    else:
        st.success("✅ Safe Order")