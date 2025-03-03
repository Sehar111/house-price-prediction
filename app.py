import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("house_price_model.pkl", "rb"))

st.title("🏡 House Price Prediction App")

# User Inputs
size = st.number_input("Enter house size (sq ft):", min_value=100, max_value=10000, step=50)
bedrooms = st.number_input("Enter number of bedrooms:", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Enter number of bathrooms:", min_value=1, max_value=10, step=1)
floors = st.number_input("Enter number of floors:", min_value=1, max_value=5, step=1)
year_built = st.number_input("Enter year built:", min_value=1800, max_value=2025, step=1)

# Categorical variables
location = st.selectbox("Location:", ["Downtown", "Suburban", "Rural"])
condition = st.selectbox("Condition of the house:", ["Excellent", "Good", "Fair"])
garage = st.selectbox("Does the house have a garage?", ["Yes", "No"])

# One-Hot Encoding: Match the same structure used during model training
location_features = [
    1 if location == "Downtown" else 0,
    1 if location == "Suburban" else 0,
    1 if location == "Rural" else 0
]  # 3 features

condition_features = [
    1 if condition == "Excellent" else 0,
    1 if condition == "Good" else 0,
    1 if condition == "Fair" else 0
]  # 3 features

garage_feature = [1 if garage == "Yes" else 0]  # 1 feature

# Combine all features (5 numerical + 3 location + 3 condition + 1 garage = 12 total)
features = np.array([[size, bedrooms, bathrooms, floors, year_built] + location_features + condition_features + garage_feature])

# Check feature count before making predictions
st.write(f"Features provided: {len(features[0])}, Model expects: {model.n_features_in_}")

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(features)
    st.success(f"🏠 Predicted House Price: ${prediction[0]:,.2f}")
