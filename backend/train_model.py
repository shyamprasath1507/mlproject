import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

def train_and_save_models(data_path="backend/historical_eco_data.csv", output_dir="backend"):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    X = df[["complexity_score", "department_count", "original_cost_usd", "change_domain"]]
    y_risk = df["risk_level"]
    y_cost = df["cost_overrun_usd"]
    
    # Preprocessing
    numeric_features = ["complexity_score", "department_count", "original_cost_usd"]
    categorical_features = ["change_domain"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )
    
    # --- Train Risk Model (Classification) ---
    print("Training Risk Classification Model...")
    # XGBoost requires target labels to be numeric (0, 1, 2)
    le = LabelEncoder()
    y_risk_encoded = le.fit_transform(y_risk)
    
    X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(X, y_risk_encoded, test_size=0.2, random_state=42)
    
    risk_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
    ])
    
    risk_pipeline.fit(X_train_risk, y_train_risk)
    
    # Save Risk Pipeline and Label Encoder
    risk_model_path = os.path.join(output_dir, "risk_model.pkl")
    label_encoder_path = os.path.join(output_dir, "risk_label_encoder.pkl")
    
    joblib.dump(risk_pipeline, risk_model_path)
    joblib.dump(le, label_encoder_path)
    print(f"Saved Risk Model to {risk_model_path}")
    
    # --- Train Cost Overrun Model (Regression) ---
    print("Training Cost Overrun Regression Model...")
    X_train_cost, X_test_cost, y_train_cost, y_test_cost = train_test_split(X, y_cost, test_size=0.2, random_state=42)
    
    cost_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(random_state=42))
    ])
    
    cost_pipeline.fit(X_train_cost, y_train_cost)
    
    # Save Cost Pipeline
    cost_model_path = os.path.join(output_dir, "cost_model.pkl")
    joblib.dump(cost_pipeline, cost_model_path)
    print(f"Saved Cost Model to {cost_model_path}")

if __name__ == "__main__":
    train_and_save_models()
