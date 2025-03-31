# House Price Predictor

This project is a machine learning model that predicts house prices based on input features. It includes data preprocessing, encoding categorical features, scaling numerical features, and training a predictive model.

## Files
- `app.py` - It is The File Which Contains the deployment part of this project which is powered by Streamlit.
- `house_price_predictor.ipynb` - Jupyter Notebook containing data preprocessing, model training, and evaluation.
- `Scaler.joblib` - Stores the fitted scaler for numerical feature normalization.
- `Encoder.joblib` - Stores the fitted encoder for categorical feature transformation.
- `Model.joblib` - Contains the trained machine learning model.
- `Encoded_cols.joblib` - Stores information about the encoded categorical columns.
- `requirements.txt` - Contains all the required libraries which are required for this project. 

## Requirements

Ensure that you have installed all the requirements which are listed in the requirements.txt file


## Usage
1. **Load the Pretrained Model and Preprocessing Objects:**
   ```python
   import joblib

   scaler = joblib.load("Scaler.joblib")
   encoder = joblib.load("Encoder.joblib")
   model = joblib.load("Model.joblib")
   encoded_cols = joblib.load("Encoded_cols.joblib")
   ```

2. **Preprocess Input Data:**
   Ensure input data is transformed using the loaded scaler and encoder.

3. **Make Predictions:**
   ```python
   prediction = model.predict(preprocessed_data)
   print("Predicted House Price:", prediction)
   ```

## Project Workflow
1. **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling numerical features.
2. **Model Training**: Training a machine learning model to predict house prices.
3. **Saving Model and Preprocessing Artifacts**: Exporting the trained model and preprocessing objects using `joblib`.


<div align="center">  
  <h2><b>Thank You</b></h2>  
</div>

