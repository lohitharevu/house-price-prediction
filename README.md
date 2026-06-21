# 🏠 House Price Prediction

A Machine Learning project that predicts house prices using housing features such as area, quality, construction year, number of rooms, and other property attributes. The project includes data preprocessing, model training, evaluation, and a Streamlit web application for interactive predictions.

---

## 📌 Overview

This project uses supervised machine learning techniques to estimate house prices based on historical housing data. The trained model can be used through an easy-to-use Streamlit interface for instant predictions.

---

## ✨ Features

- Data preprocessing and feature engineering
- Model training and evaluation
- Scaler and feature persistence using Pickle
- Interactive Streamlit dashboard
- Single house price prediction
- Performance metrics tracking
- Modular project structure

---

## 📂 Project Structure

```text
HOUSE/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── test_data.csv
│
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── results/
│   └── metrics.csv
│
├── utils/
│   ├── preprocess.py
│   └── prediction.py
│
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Joblib
- Streamlit
- Matplotlib

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Training the Model

Run the training script:

```bash
python train.py
```

This will:

- Load and preprocess the dataset
- Train the machine learning model
- Save the trained model
- Save scaler and feature columns
- Generate evaluation metrics

Generated files:

```text
models/model.pkl
models/scaler.pkl
models/feature_columns.pkl
results/metrics.csv
```

---

## ▶️ Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal to access the dashboard.

---

## 📊 Dataset

The project uses housing data containing various property-related features.

Example attributes include:

- Lot Area
- Overall Quality
- Overall Condition
- Year Built
- Living Area
- Number of Bedrooms
- Bathrooms
- Garage Capacity

Target Variable:

```text
SalePrice
```

---

## 📈 Model Evaluation

Model performance is measured using:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

The evaluation results are automatically stored in:

```text
results/metrics.csv
```

---

## 🎯 Prediction Workflow

1. User enters house details.
2. Input data is preprocessed.
3. Features are scaled using the saved scaler.
4. Trained model predicts house price.
5. Predicted value is displayed on the dashboard.

---

## 🔮 Future Improvements

- XGBoost and LightGBM models
- Hyperparameter tuning
- Feature importance visualization
- Batch prediction via CSV upload
- Deployment on Streamlit Cloud
- Advanced analytics dashboard

---

## 👨‍💻 Author

Lohitha Revu

---

⭐ If you found this project helpful, consider giving it a star on GitHub.
