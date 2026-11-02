import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Predictor", page_icon="📉")
st.title("Customer Churn Predictor")
st.write("Enter a customer's details to see the chance they will leave.")


@st.cache_resource          # load the model once, not on every click
def load():
    return joblib.load("churn_model.joblib")


bundle = load()
model, threshold = bundle["model"], bundle["threshold"]

# TotalCharges is calculated below, so we don't ask the user for it
fields = [name for name in bundle["columns"] if name != "TotalCharges"]

# Build one input per feature: dropdown for text, number box for numbers
row = {}
left, right = st.columns(2)
for i, name in enumerate(fields):
    box = left if i % 2 == 0 else right
    if name in bundle["cat_options"]:
        row[name] = box.selectbox(name, bundle["cat_options"][name])
    else:
        lo, hi, med = bundle["num_ranges"][name]
        row[name] = box.number_input(name, min_value=lo, max_value=hi, value=med)

if st.button("Predict"):
    # Total charges = months with the company x monthly charge
    row["TotalCharges"] = row["tenure"] * row["MonthlyCharges"]

    prob = model.predict_proba(pd.DataFrame([row]))[0, 1]
    st.metric("Churn probability", f"{prob:.0%}")

    if prob >= threshold:
        st.error("High risk: consider a retention offer.")
    else:
        st.success("Lower risk of leaving.")