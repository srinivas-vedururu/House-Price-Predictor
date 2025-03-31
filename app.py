import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model and preprocessors (Assuming these are pre-trained and available)
scaler = joblib.load("loaded_joblib_files\Scaler.joblib")
encoder = joblib.load("loaded_joblib_files\Encoder.joblib")
model = joblib.load("loaded_joblib_files\Model.joblib")

# Streamlit UI
st.title("House Price Predictor")
st.markdown("### Enter the details of your desired house:")

# User Inputs
square_feet = st.number_input("Square Feet", min_value=100, step=50, value=1000)
bedrooms = st.number_input("Bedrooms", min_value=1, step=1, value=2)
bathrooms = st.number_input("Bathrooms", min_value=1, step=1, value=1)
neighborhood = st.selectbox("Neighborhood", ["Rural", "Suburb", "Urban"])
year_built = st.number_input("Year Built", min_value=1800, max_value=2025, step=1, value=2000)

# Calculate additional features
age = 2025 - year_built
sqft_per_bedroom = square_feet / bedrooms if bedrooms > 0 else square_feet
bed_bath_ratio = bedrooms / (bathrooms if bathrooms > 0 else 1)
log_square_feet = np.log1p(square_feet)
total_rooms = bedrooms + bathrooms
bed_bath_interaction = bedrooms * bathrooms

# Define bins for home size categories
max_sqft = 5000  # Placeholder max square feet
bins = [0, 1000, 2000, 3000, 4000, max_sqft]
labels = ['Small', 'Medium', 'Large', 'XL', 'Mansion']
home_size_category = pd.cut([square_feet], bins=bins, labels=labels, include_lowest=True)[0]

# Create input dataframe
single_input = pd.DataFrame({
    'SquareFeet': [square_feet],
    'Bedrooms': [bedrooms],
    'Bathrooms': [bathrooms],
    'Neighborhood': [neighborhood],
    'YearBuilt': [year_built],
    'Age': [age],
    'SqFt_Per_Bedroom': [sqft_per_bedroom],
    'Bed_Bath_Ratio': [bed_bath_ratio],
    'Log_SquareFeet': [log_square_feet],
    'Total_Rooms': [total_rooms],
    'Bed_Bath_Interaction': [bed_bath_interaction],
    'Home_Size_Category': [home_size_category]
})

# Assume numerical and categorical preprocessing is already fitted
num_cols = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'YearBuilt', 'Age', 'SqFt_Per_Bedroom', 'Bed_Bath_Ratio', 'Log_SquareFeet', 'Total_Rooms', 'Bed_Bath_Interaction']
cat_cols = ['Neighborhood', 'Home_Size_Category']

single_input[num_cols] = scaler.transform(single_input[num_cols])
encoded_cols = joblib.load("loaded_joblib_files\Encoded_cols.joblib")
single_input[encoded_cols] = encoder.transform(single_input[cat_cols])
single_input = single_input[num_cols+encoded_cols]

# Predict
if st.button("Predict Price"):
    prediction = model.predict(single_input)
    st.success(f"The estimated house price is: €{round(prediction[0],3 )}")