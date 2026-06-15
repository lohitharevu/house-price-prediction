import streamlit as st
import pandas as pd

from utils.prediction import predict_price, predict_batch, get_model_info
from utils.preprocess import feature_engineering, preprocess_pipeline

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

if "page" not in st.session_state:
    st.session_state.page = "Home"

st.sidebar.title("🏠 Navigation")

if st.sidebar.button("🏡 Home"):
    st.session_state.page = "Home"

if st.sidebar.button("🔮 Single Prediction"):
    st.session_state.page = "Single"

if st.sidebar.button("📂 Batch Prediction"):
    st.session_state.page = "Batch"

if st.sidebar.button("🤖 Model Info"):
    st.session_state.page = "Model"

if st.session_state.page == "Home":

    st.title("🏠 House Price Prediction App")

    st.write("""
    Welcome 👋  

    This app predicts house prices using Machine Learning.

    Features:
    - Single house prediction
    - Batch CSV prediction
    - Model information dashboard
    """)

    st.info("Use the navigation buttons on the left sidebar.")

elif st.session_state.page == "Single":

    st.title("🏡 Single House Price Prediction")

    st.subheader("Enter House Details")

    col1, col2 = st.columns(2)

    with col1:
        lot_area = st.number_input("Lot Area", min_value=1000, value=5000)
        overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
        year_built = st.number_input("Year Built", min_value=1900, max_value=2026, value=2000)

    with col2:
        gr_liv_area = st.number_input("Living Area (sq ft)", min_value=300, value=1500)
        full_bath = st.number_input("Full Bathrooms", min_value=0, value=2)
        half_bath = st.number_input("Half Bathrooms", min_value=0, value=1)

    if st.button("Predict Price 💰"):

        input_data = {
            "LotArea": lot_area,
            "OverallQual": overall_qual,
            "YearBuilt": year_built,
            "GrLivArea": gr_liv_area,
            "FullBath": full_bath,
            "HalfBath": half_bath
        }

        df = pd.DataFrame([input_data])

        # Feature engineering
        df = feature_engineering(df)

        # Prediction
        result = predict_price(df.iloc[0].to_dict())

        st.success(f"🏠 Predicted House Price: ${result['predicted_price']}")

elif st.session_state.page == "Batch":

    st.title("📂 Batch Prediction")

    st.write("Upload a CSV file for predictions")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:

        df = pd.read_csv(file)

        st.subheader("Preview Data")
        st.dataframe(df.head())

        if st.button("Run Prediction 🚀"):

            # preprocess + predict
            df_processed = preprocess_pipeline(df)
            result_df = predict_batch(df_processed)

            st.success("Prediction Completed!")

            st.subheader("Results")
            st.dataframe(result_df)

            csv = result_df.to_csv(index=False)

            st.download_button(
                "Download Predictions 📥",
                csv,
                "house_price_predictions.csv",
                "text/csv"
            )

elif st.session_state.page == "Model":

    st.title("🤖 Model Information Dashboard")

    info = get_model_info()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Model Type", info["model_type"])

    with col2:
        st.metric("Feature Count", info["features_count"])

    st.info("""
    This model is trained using regression algorithms such as:

    - Linear Regression
    - Random Forest Regressor

    The best performing model is selected automatically during training.
    """)

    st.success("Model is ready for prediction 🚀")