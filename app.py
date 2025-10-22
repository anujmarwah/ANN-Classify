import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle

#Load the pre-trained model
model=tf.keras.models.load_model('model.h5')

### load the trained model,scaler pickles and test data from previous steps ###
model=tf.keras.models.load_model('model.h5')
with open('scaler.pkl','rb') as f:
    scaler=pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('onehot_encoder_geo.pkl','rb') as f:
    onehot_encoder_geo = pickle.load(f)

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn probability.")
#Input fields
CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
Geography = st.selectbox("Geography", options=["France", "Germany", "Spain"])
Gender = st.selectbox("Gender", options=["Male", "Female"])
Age = st.slider("Age", min_value=18, max_value=92, value=30)
Tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)
Balance = st.number_input("Balance", min_value=0.0, value=10000.0)
NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
HasCrCard = st.selectbox("Has Credit Card", options=[0, 1])
IsActiveMember = st.selectbox("Is Active Member", options=[0, 1])
EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
input_data = pd.DataFrame({
    'CreditScore': [CreditScore],
    'Gender': [label_encoder_gender.transform([Gender])[0]],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
})

geo_encoded = onehot_encoder_geo.transform([[Geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_data)
# Ensure all expected columns are present
prediction = model.predict(input_data_scaled)
churn_probability = prediction[0][0]
st.write(f"Churn Probability: {churn_probability:.2f}")
if churn_probability > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is unlikely to churn.")