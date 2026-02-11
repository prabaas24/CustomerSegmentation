import streamlit as st
import pandas as pd
import pickle
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



st.title("Customer Segmentation Prediction App")

st.write("Enter customer details to predict cluster")



income = st.number_input("Income", 0, 200000, 50000)
age = st.number_input("Age", 18, 100, 40)
spending = st.number_input("Total Spending", 0, 3000, 500)
children = st.number_input("Children Count", 0, 5, 1)
recency = st.number_input("Recency", 0, 100, 50)



if st.button("Predict Cluster"):

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




   







