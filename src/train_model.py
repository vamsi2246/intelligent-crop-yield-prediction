"""
train_model.py

Trains multiple ML models on the crop yield dataset, evaluates them,
selects the best one, and saves all artifacts to models/.

Dataset: https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset

Usage:
    python src/train_model.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = "data/yield_df.csv"
MODELS_DIR = "models"
TARGET_COLUMN = "hg/ha_yield"
RANDOM_STATE = 42
TEST_SIZE = 0.2


print("=" * 60)
print("  CROP YIELD PREDICTION — MODEL TRAINING PIPELINE")
print("=" * 60)

print("\nLoading dataset...")
data = pd.read_csv(DATA_PATH)
print(f"  Shape: {data.shape[0]} rows x {data.shape[1]} columns")
print(f"  Columns: {list(data.columns)}")

if TARGET_COLUMN not in data.columns:
    print(f"\nERROR: Target column '{TARGET_COLUMN}' not found!")
    print(f"  Available columns: {list(data.columns)}")
    exit(1)

unnamed_cols = [c for c in data.columns if "Unnamed" in c]
if unnamed_cols:
    data = data.drop(columns=unnamed_cols)
    print(f"  Dropped index columns: {unnamed_cols}")


print("\nPreprocessing data...")

missing = data.isnull().sum()
if missing.sum() > 0:
    print("  Missing values found:")
    for col, count in missing[missing > 0].items():
        print(f"    {col}: {count}")

data = data.dropna()
print(f"  Rows after cleaning: {data.shape[0]}")

categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
numeric_feature_cols = [
    c for c in data.select_dtypes(include=[np.number]).columns.tolist()
    if c != TARGET_COLUMN
]

print(f"  Categorical features: {categorical_cols}")
print(f"  Numeric features: {numeric_feature_cols}")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
    print(f"  Encoded '{col}': {len(le.classes_)} unique values")

X = data.drop(columns=[TARGET_COLUMN])
y = data[TARGET_COLUMN]

feature_names = list(X.columns)

scaler = StandardScaler()
X[numeric_feature_cols] = scaler.fit_transform(X[numeric_feature_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"\n  Train set: {X_train.shape[0]} samples")
print(f"  Test set:  {X_test.shape[0]} samples")


print("\nTraining models...\n")

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=20,
        max_depth=10,
        random_state=RANDOM_STATE
    ),
}

results = {}

for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        "model": model,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    }
    print(f"    MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")


print("\n" + "=" * 60)
print("  MODEL COMPARISON")
print("=" * 60)
print(f"\n  {'Model':<25} {'MAE':>10} {'RMSE':>12} {'R2':>10}")
print("  " + "-" * 57)

for name, r in results.items():
    print(f"  {name:<25} {r['MAE']:>10.4f} {r['RMSE']:>12.4f} {r['R2']:>10.4f}")

best_name = max(results, key=lambda k: results[k]["R2"])
best_model = results[best_name]["model"]
print(f"\n  Best model: {best_name} (R2 = {results[best_name]['R2']:.4f})")


print("\nFeature Importance (from best model):")

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    importance_dict = dict(zip(feature_names, importances.tolist()))
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_features:
        bar = "#" * int(imp * 50)
        print(f"  {feat:<35} {imp:.4f}  {bar}")
elif hasattr(best_model, "coef_"):
    coefs = np.abs(best_model.coef_)
    importance_dict = dict(zip(feature_names, coefs.tolist()))
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_features:
        print(f"  {feat:<35} {imp:.4f}")
else:
    importance_dict = {}
    print("  Feature importance not available for this model type.")


print(f"\nSaving artifacts to {MODELS_DIR}/...")
os.makedirs(MODELS_DIR, exist_ok=True)

model_path = os.path.join(MODELS_DIR, "crop_yield_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

encoders_path = os.path.join(MODELS_DIR, "label_encoders.pkl")
with open(encoders_path, "wb") as f:
    pickle.dump(label_encoders, f)

scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

columns_path = os.path.join(MODELS_DIR, "model_columns.pkl")
with open(columns_path, "wb") as f:
    pickle.dump(feature_names, f)

metrics_to_save = {
    name: {k: v for k, v in r.items() if k != "model"}
    for name, r in results.items()
}
metrics_to_save["best_model"] = best_name
metrics_to_save["feature_importance"] = importance_dict

metrics_path = os.path.join(MODELS_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics_to_save, f, indent=2)

model_size = os.path.getsize(model_path) / (1024 * 1024)
print(f"\n  crop_yield_model.pkl  — {model_size:.2f} MB")
print(f"  label_encoders.pkl   — {os.path.getsize(encoders_path) / 1024:.2f} KB")
print(f"  scaler.pkl           — {os.path.getsize(scaler_path) / 1024:.2f} KB")
print(f"  model_columns.pkl    — {os.path.getsize(columns_path) / 1024:.2f} KB")
print(f"  metrics.json         — {os.path.getsize(metrics_path) / 1024:.2f} KB")

if model_size < 50:
    print("\n  Model file is under 50 MB — safe for GitHub.")
else:
    print("\n  WARNING: Model exceeds 50 MB — consider reducing n_estimators or max_depth.")

print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("  Run the app: streamlit run app.py")
print("=" * 60)
