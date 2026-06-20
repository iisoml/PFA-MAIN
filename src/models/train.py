import mlflow
import pandas as pd
import numpy as np
import pickle
import os
import mlflow.sklearn
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

CATEGORICAL_FEATURES = ["labname", "gender", "age", "unittype", "recent_diagnosis"]
NUMERIC_FEATURES = ["result_time", "validation_time", "admissionweight", "lab_workload_last_hour"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "turnaround_time_mins"

def train_model(df: pd.DataFrame, target_column: str = TARGET):
    X = df[ALL_FEATURES].copy()
    y = df[target_column].copy()
    
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("Unknown")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ("num", numeric_transformer, NUMERIC_FEATURES),
    ])
    
    model = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, tree_method="hist"
    )
    
    pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", model)])
    
    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_param("n_estimators", 500)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(pipeline, "model")
        
        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")
        
        os.makedirs("models", exist_ok=True)
        with open("models/pipeline.pkl", "wb") as f:
            pickle.dump(pipeline, f)
        
        print(f"Model trained. MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    return pipeline

if __name__ == "__main__":
    df = pd.read_csv("lab_pred.csv")
    train_model(df)