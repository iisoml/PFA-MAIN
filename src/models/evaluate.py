from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a regression model on test data.

    Args:
        model: Trained regression model (e.g., sklearn Pipeline with XGBRegressor).
        X_test: Test features (DataFrame or array-like).
        y_test: True target values (continuous).

    Returns:
        dict: Dictionary containing MAE, RMSE, and R² scores.
    """
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"Regression Metrics:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    return {"mae": mae, "rmse": rmse, "r2": r2}