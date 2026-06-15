import pickle
import pandas as pd
import numpy as np

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

def preprocess_input(input_df):
    """
    Align input data with training features
    """

    # One-hot encoding (same as training)
    input_df = pd.get_dummies(input_df)

    # Add missing columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure correct column order
    input_df = input_df[feature_columns]

    return input_df

def predict_price(input_data):
    """
    Predict house price for single input
    input_data: dict
    """

    df = pd.DataFrame([input_data])

    processed_df = preprocess_input(df)

    scaled_df = scaler.transform(processed_df)

    prediction = model.predict(scaled_df)[0]

    return {
        "predicted_price": round(float(prediction), 2)
    }

def predict_batch(dataframe):
    """
    Predict multiple house prices
    dataframe: pandas DataFrame
    """

    processed_df = preprocess_input(dataframe)

    scaled_df = scaler.transform(processed_df)

    predictions = model.predict(scaled_df)

    dataframe["Predicted_Price"] = predictions

    return dataframe

def get_model_info():
    return {
        "model_type": str(type(model).__name__),
        "features_count": len(feature_columns)
    }