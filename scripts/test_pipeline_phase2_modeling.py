# train_model.py (alternative, more modular)
import pickle
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def train_model(X_train, X_test, y_train, y_test, best_params=None):
    if best_params is None:
        best_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "tree_method": "hist",
            "n_jobs": -1,
        }
    
    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    os.makedirs("models", exist_ok=True)
    with open("models/xgb_regressor.pkl", "wb") as f:
        pickle.dump(model, f)
    
    return model, {"mae": mae, "rmse": rmse, "r2": r2}