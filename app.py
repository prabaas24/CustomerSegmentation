import streamlit as st
import pandas as pd
import pickle


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


st.title("Customer Segmentation Prediction App")
st.write("Enter customer details to predict cluster")

income = st.number_input("Income", min_value=0, max_value=200000, value=50000)
age = st.number_input("Age", min_value=18, max_value=100, value=40)
spending = st.number_input("Total Spending", min_value=0, max_value=5000, value=500)
children = st.number_input("Children Count", min_value=0, max_value=5, value=1)
recency = st.number_input("Recency (Days Since Last Purchase)", min_value=0, max_value=100, value=50)


if st.button("Predict Cluster"):

    input_data = pd.DataFrame([[income, age, spending, children, recency]],
                              columns=features)

    scaled_input = scaler.transform(input_data)

    cluster_id = model.predict(scaled_input)[0]

    segment_name, segment_desc = segment_map[cluster_id]

    st.success(f"Segment: {segment_name}")
    st.write(segment_desc)


