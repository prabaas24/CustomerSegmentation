import streamlit as st
import pandas as pd
import pickle
<<<<<<< HEAD


model = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("model_features.pkl", "rb"))


segment_map = {
    0: (
        "Low Value Young Customers",
        "Low income, low spending, price sensitive customers with moderate engagement."
    ),
    1: (
        "Dormant Mature Customers",
        "Older customers with high recency. Previously active but at risk of churn."
    ),
    2: (
        "Active Mature Loyal Customers",
        "Recently active customers with stable income and consistent spending."
    ),
    3: (
        "Premium High Value Customers",
        "High income, high spending, low price sensitivity. Prime lifetime value segment."
    )
}

=======
import numpy as np

model = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
with open("model_features.pkl", "rb") as f:
    model_features = pickle.load(f)

segment_map = {
    0: ("Moderate Value Family Customers",
        "Budget conscious, family spending driven"),
        
    1: ("Stable Loyal Customers",
        "Consistent spending, mature customer base"),
        
    2: ("Premium High Value Customers",
        "High income, high spending, low price sensitivity"),
        
    3: ("Low Value Dormant Customers",
        "Low engagement, price sensitive")
}


>>>>>>> 50d4e02c9d6ee76bce8a1db78f9f8d60b582e513

st.title("Customer Segmentation Prediction App")
st.write("Enter customer details to predict cluster")

income = st.number_input("Income", min_value=0, max_value=200000, value=50000)
age = st.number_input("Age", min_value=18, max_value=100, value=40)
spending = st.number_input("Total Spending", min_value=0, max_value=5000, value=500)
children = st.number_input("Children Count", min_value=0, max_value=5, value=1)
recency = st.number_input("Recency (Days Since Last Purchase)", min_value=0, max_value=100, value=50)


if st.button("Predict Cluster"):

<<<<<<< HEAD
    input_data = pd.DataFrame([[income, age, spending, children, recency]],
                              columns=features)

    scaled_input = scaler.transform(input_data)

    cluster_id = model.predict(scaled_input)[0]

    segment_name, segment_desc = segment_map[cluster_id]

    st.success(f"Segment: {segment_name}")
    st.write(segment_desc)
=======
    input_dict = {col: 0 for col in model_features}

    input_dict["Income"] = income
    input_dict["Age"] = age
    input_dict["Recency"] = recency

    if "Total_Spending" in model_features:
        input_dict["Total_Spending"] = spending

    if "Children_Count" in model_features:
        input_dict["Children_Count"] = children

    input_df = pd.DataFrame([input_dict])

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)
    cluster_id = prediction[0]

    st.write("Predicted Cluster:", cluster_id)

    segment_name, segment_desc = segment_map[cluster_id]
    st.success(f"Segment: {segment_name}")
    st.write(segment_desc)




   







>>>>>>> 50d4e02c9d6ee76bce8a1db78f9f8d60b582e513
