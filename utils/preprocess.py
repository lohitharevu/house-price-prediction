import pandas as pd

def feature_engineering(df):
    """
    Create additional useful features
    """

    df = df.copy()

    # Total bathrooms
    if "FullBath" in df.columns and "HalfBath" in df.columns:
        df["TotalBath"] = df["FullBath"] + (0.5 * df["HalfBath"])

    # House age
    if "YearBuilt" in df.columns:
        df["HouseAge"] = 2026 - df["YearBuilt"]

    return df

def handle_missing_values(df):
    """
    Fill missing values safely
    """

    df = df.copy()

    # Numeric columns → median
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Categorical columns → mode
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

def clean_data(df):
    """
    Basic cleaning
    """

    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    return df

def preprocess_pipeline(df):
    """
    Complete preprocessing pipeline
    """

    df = clean_data(df)
    df = handle_missing_values(df)
    df = feature_engineering(df)

    return df

def encode_features(df):
    """
    One-hot encoding
    """

    return pd.get_dummies(df)