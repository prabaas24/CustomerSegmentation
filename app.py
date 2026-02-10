import streamlit as st
import pandas as pd
import pickle
import numpy as np

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


model = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("model_features.pkl","rb"))


st.title("Customer Segmentation Prediction App")

st.write("Enter customer details to predict cluster")



income = st.number_input("Income", 0, 200000, 50000)
age = st.number_input("Age", 18, 100, 40)
spending = st.number_input("Total Spending", 0, 3000, 500)
children = st.number_input("Children Count", 0, 5, 1)
recency = st.number_input("Recency", 0, 100, 50)



if st.button("Predict Cluster"):
    input_df = pd.DataFrame(columns=features)
    input_df.loc[0] = 0
    input_df['Income'] = income
    input_df['Age'] = age
    input_df['Total_Spending'] = spending
    input_df['Children_Count'] = children
    input_df['Recency'] = recency


    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    segment_name, segment_desc = segment_map[prediction[0]]
    st.success(f"Segment: {segment_name}")
    st.write(segment_desc)
 
   







