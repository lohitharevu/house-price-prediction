import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/train.csv")

print("Dataset Shape:", df.shape)

y = df["SalePrice"]

# Drop target + ID
X = df.drop(["SalePrice", "Id"], axis=1)

# Fill numerical columns with median
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

# Fill categorical columns with mode
cat_cols = X.select_dtypes(include=["object"]).columns
for col in cat_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
}


best_model = None
best_score = -np.inf
best_model_name = ""

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{name} Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE : {mae:.2f}")
    print(f"R2  : {r2:.4f}")

    if r2 > best_score:
        best_score = r2
        best_model = model
        best_model_name = name

with open("models/model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save feature columns (VERY IMPORTANT for app.py)
with open("models/feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns, f)

results = {
    "Best Model": best_model_name,
    "Best R2 Score": best_score
}

pd.DataFrame([results]).to_csv("results/metrics.csv", index=False)

print("\n==============================")
print("Training Completed Successfully")
print("Best Model:", best_model_name)
print("Best R2 Score:", best_score)
print("==============================")