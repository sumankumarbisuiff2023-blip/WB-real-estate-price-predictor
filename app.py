import streamlit as st
import pandas as pd
import joblib

# Load the saved upgraded model pipeline
model = joblib.load('models/house_price_model.pkl')

st.title("🏠 West Bengal Real Estate Price Predictor")
st.write("Interactive ML Web App built with scikit-learn and Streamlit.")

# Sidebar or main inputs for property features
st.header("Enter Property Details:")

sqft = st.number_input("Square Feet Area (SqftArea)", min_value=200, max_value=10000, value=1200)
bedrooms = st.selectbox("Number of Bedrooms (BHK)", [1, 2, 3, 4, 5])

# List of common locations from your dataset or general West Bengal regions
location_options = [
    'New Town', 'Salt Lake', 'Garia', 'Park Street', 'Behala', 
    'Howrah', 'Jadavpur', 'Tollygunge', 'Baruipur', 'Baranagar', 'Gariahat'
]
location = st.selectbox("Location", location_options)

# Prediction button
if st.button("Predict Property Price"):
    # Create a dataframe for the model input
    input_df = pd.DataFrame({
        'SqftArea': [sqft],
        'Bedrooms': [bedrooms],
        'Location': [location]
    })

    # Make prediction using our upgraded pipeline
    prediction = model.predict(input_df)[0]

    # Display the result clearly
    st.success(f"Estimated Market Price: ₹ {prediction:,.2f} (~{prediction / 100000:.2f} Lakhs)")
