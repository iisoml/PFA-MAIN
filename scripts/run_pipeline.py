#!/usr/bin/env python3
"""
Runs sequentially: load → validate → preprocess → feature engineering → train → evaluate
"""

import os
import sys
import time
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, f1_score,
)
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, LabelEncoder,
)
from xgboost import XGBRegressor

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow not found — MLP classifier will be skipped.")

# === Fix import path for local modules ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local modules
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

def validate_lab_data(df):
    """
    Basic data quality validation for lab prediction data.
    Returns (is_valid, list of issues).
    """
    issues = []
    
    required_cols = [
        "labid", "labname", "result_time", "validation_time",
        "turnaround_time_mins", "gender", "age", "unittype",
        "admissionweight", "recent_diagnosis", "lab_workload_last_hour"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
    
    if "turnaround_time_mins" in df.columns:
        if df["turnaround_time_mins"].isna().sum() > 0:
            issues.append("Target has NaN values")
        if (df["turnaround_time_mins"] <= 0).any():
            issues.append("Target has non-positive values")
    
    return len(issues) == 0, issues


def main(args):
    """
    Main training pipeline for lab turnaround time prediction.
    """
    
    # === MLflow Setup ===
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_path = args.mlflow_uri or f"file://{project_root}/mlruns"
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        # === Log configuration ===
        mlflow.log_param("model", "xgboost_regressor")
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("learning_rate", args.learning_rate)

        # === STAGE 1: Data Loading ===
        print("Loading data...")
        df = load_data(args.input)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # === STAGE 2: Data Validation ===
        print("Validating data quality...")
        is_valid, issues = validate_lab_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))

        if not is_valid:
            mlflow.log_text(json.dumps(issues, indent=2), artifact_file="validation_issues.json")
            raise ValueError(f"Data validation failed: {issues}")
        else:
            print("Data validation passed.")

        # === STAGE 3: Preprocessing ===
        print("Preprocessing data...")
        df = preprocess_data(df, target_column=args.target)

        processed_path = os.path.join(project_root, "data", "processed", "lab_pred_processed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"Processed dataset saved to {processed_path} | Shape: {df.shape}")

        # === STAGE 4: Feature Engineering ===
        print("Building features...")
        target = args.target
        
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        df_enc, selected_features = build_features(df, target_column=target)
        
        # Convert boolean columns to integers for XGBoost compatibility
        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)
        
        print(f"Feature engineering completed: {len(selected_features)} features selected")

        # === Save Feature Metadata & Preprocessing Artifacts ===
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        feature_cols = selected_features  # Already excludes target from build_features

        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

        # --- Fit preprocessing transformers on training data ---
        # These constants must match inference.py exactly.
        XGB_NUM_FEATURES = [
            "age", "admissionweight", "lab_workload_last_hour",
            "result_hour", "result_weekday", "result_year", "result_month", "result_day",
        ]
        XGB_CAT_FEATURES = ["labname", "gender", "unittype", "recent_diagnosis"]
        PERIOD_LABELS    = ["apres_midi", "matin", "nuit", "soir"]

        medians = df_enc[XGB_NUM_FEATURES].median()

        xgb_scaler = StandardScaler()
        xgb_scaler.fit(df_enc[XGB_NUM_FEATURES].fillna(medians).values)

        try:
            xgb_onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:                          # scikit-learn < 1.2
            xgb_onehot = OneHotEncoder(handle_unknown="ignore", sparse=True)
        xgb_onehot.fit(df_enc[XGB_CAT_FEATURES].fillna("unknown").astype(str))

        label_encoder = LabelEncoder()
        label_encoder.fit(PERIOD_LABELS)

        preprocessing_artifact = {
            # XGBoost serving
            "feature_columns": feature_cols,
            "target":          target,
            "num_features":    XGB_NUM_FEATURES,
            "cat_features":    XGB_CAT_FEATURES,
            "period_labels":   PERIOD_LABELS,
            "scaler":          xgb_scaler,
            "onehot":          xgb_onehot,
            "label_encoder":   label_encoder,
            "medians":         medians,
        }
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
        print(f"Saved {len(feature_cols)} feature columns and fitted preprocessing transformers")

        # === STAGE 5: Train/Test Split ===
        print("Splitting data...")
        X = df_enc[feature_cols]
        y = df_enc[target]
        
        # No stratify for regression (continuous target)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            random_state=42
        )
        print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

        # === STAGE 6: Model Training ===
        print("Training XGBoost regressor...")
        
        model = XGBRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            tree_method="hist",
        )

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)
        print(f"Model trained in {train_time:.2f} seconds")

        # === STAGE 7: Model Evaluation (Regression Metrics) ===
        print("Evaluating model performance...")
        
        t1 = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - t1
        
        # Regression metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("pred_time", pred_time)
        
        print(f"   Model Performance:")
        print(f"   MAE:  {mae:.2f} minutes")
        print(f"   RMSE: {rmse:.2f} minutes")
        print(f"   R²:   {r2:.4f}")

        # === Residual Analysis ===
        residuals = y_test - y_pred
        mlflow.log_metric("residual_mean", residuals.mean())
        mlflow.log_metric("residual_std", residuals.std())
        
        # Log prediction vs actual scatter data for visualization
        pred_df = pd.DataFrame({
            "actual": y_test.values,
            "predicted": y_pred,
            "residual": residuals.values
        })
        pred_df.to_csv(os.path.join(artifacts_dir, "predictions.csv"), index=False)
        mlflow.log_artifact(os.path.join(artifacts_dir, "predictions.csv"))

        # === Feature Importance ===
        importance = pd.Series(model.feature_importances_, index=feature_cols)
        top_features = importance.nlargest(20)
        mlflow.log_text(top_features.to_string(), artifact_file="feature_importance.txt")
        print(f"\n Top 10 Important Features:")
        print(top_features.head(10).to_string())

        # === STAGE 8: Model Serialization ===
        print("Saving model to MLflow...")
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        # Also save locally
        model_path = os.path.join(artifacts_dir, "model.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        print("Model saved to MLflow")

        # ================================================================
        # === STAGE 9: MLP Classifier — turnaround period prediction ===
        # ================================================================
        # The MLP predicts which time-of-day period (matin / apres_midi /
        # soir / nuit) the lab result will be VALIDATED in.
        # Target = categorize_hour(validation_hour), matching the notebook.
        #
        # Feature set: 5 numeric columns only (age, admissionweight,
        # lab_workload_last_hour, result_hour, result_weekday).
        # This matches the original notebook (Cell 41) and the MLP weight
        # matrix shape (N_samples × 5 → Dense(64) → Dense(32) → Dense(4)).
        # ================================================================
        if TF_AVAILABLE:
            print("\nTraining MLP classifier (turnaround period)...")

            # --- Build classification target ---
            def _categorize_hour(h):
                if 6 <= h < 12:   return "matin"
                if 12 <= h < 18:  return "apres_midi"
                if 18 <= h < 24:  return "soir"
                return "nuit"

            # Estimate validation hour from result_hour + predicted turnaround
            # We use the XGBoost model we just trained to get those predictions.
            X_all_xgb = df_enc[feature_cols]
            tat_all   = model.predict(X_all_xgb)

            result_hours     = df_enc["result_hour"].values if "result_hour" in df_enc.columns else np.zeros(len(df_enc))
            validation_hours = ((result_hours + tat_all / 60) % 24).astype(int)
            turnaround_cat   = pd.Series(validation_hours).apply(_categorize_hour)

            # --- MLP features (5 numeric columns only) ---
            MLP_FEATURES = [
                "age", "admissionweight", "lab_workload_last_hour",
                "result_hour", "result_weekday",
            ]
            # Use df_enc which already has result_hour / result_weekday from build_features
            X_clf = df_enc[MLP_FEATURES].copy()
            for col in MLP_FEATURES:
                X_clf[col] = pd.to_numeric(X_clf[col], errors="coerce")
            mlp_medians = X_clf.median()
            X_clf = X_clf.fillna(mlp_medians)

            # --- Encode target ---
            mlp_label_encoder = LabelEncoder()
            mlp_label_encoder.fit(sorted(turnaround_cat.unique()))
            y_clf = mlp_label_encoder.transform(turnaround_cat)
            num_classes = len(mlp_label_encoder.classes_)
            print(f"   MLP classes: {mlp_label_encoder.classes_.tolist()}")

            # --- Scale features ---
            mlp_scaler = StandardScaler()
            X_clf_scaled = mlp_scaler.fit_transform(X_clf.values)

            # --- Train/test split ---
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_clf_scaled, y_clf, test_size=0.2, random_state=42
            )
            print(f"   Train: {X_tr.shape[0]} | Test: {X_te.shape[0]}")

            # --- Build MLP (matches notebook Cell 42 architecture) ---
            mlp_model = models.Sequential([
                layers.Dense(64, activation="relu", input_shape=(X_tr.shape[1],)),
                layers.Dropout(0.3),
                layers.Dense(32, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(num_classes, activation="softmax"),
            ])
            mlp_model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            early_stop = EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
            mlp_history = mlp_model.fit(
                X_tr, y_tr,
                validation_data=(X_te, y_te),
                epochs=30, batch_size=32,
                callbacks=[early_stop], verbose=0,
            )

            # --- Evaluate ---
            _, mlp_acc = mlp_model.evaluate(X_te, y_te, verbose=0)
            y_pred_clf = mlp_model.predict(X_te, verbose=0).argmax(axis=1)
            mlp_f1 = f1_score(y_te, y_pred_clf, average="weighted")

            mlflow.log_metric("mlp_accuracy", mlp_acc)
            mlflow.log_metric("mlp_f1_weighted", mlp_f1)
            print(f"   MLP Accuracy : {mlp_acc:.2%}")
            print(f"   MLP F1 (weighted): {mlp_f1:.4f}")
            print(classification_report(
                y_te, y_pred_clf,
                target_names=mlp_label_encoder.classes_,
            ))

            # --- Save MLP model ---
            mlp_path = os.path.join(project_root, "mlp_classification.h5")
            mlp_model.save(mlp_path)
            mlflow.log_artifact(mlp_path)
            print(f"   MLP saved → {mlp_path}")

            # --- Save MLP preprocessing alongside main artifact ---
            preprocessing_artifact["mlp_features"]      = MLP_FEATURES
            preprocessing_artifact["mlp_scaler"]        = mlp_scaler
            preprocessing_artifact["mlp_label_encoder"] = mlp_label_encoder
            preprocessing_artifact["mlp_medians"]       = mlp_medians
            joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
            mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
            print("   Updated preprocessing.pkl with MLP transformers")

        else:
            print("\nSkipping MLP training (TensorFlow not installed).")
            print("Install it with:  pip install tensorflow")

        # === Performance Summary ===
        print(f"\n Performance Summary:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Inference time: {pred_time:.4f}s")
        print(f"   Samples per second: {len(X_test)/pred_time:.0f}")
        
        print(f"\n Residual Statistics:")
        print(f"   Mean residual: {residuals.mean():.2f} mins")
        print(f"   Std residual:  {residuals.std():.2f} mins")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run lab TAT prediction pipeline with XGBoost + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., lab_pred.csv)")
    p.add_argument("--target", type=str, default="turnaround_time_mins")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--n_estimators", type=int, default=500)
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--experiment", type=str, default="Lab Turnaround Time")
    p.add_argument("--mlflow_uri", type=str, default=None,
                   help="override MLflow tracking URI")

    args = p.parse_args()
    main(args)

"""
# Run the pipeline:
python run_pipeline.py --input lab_pred.csv
"""