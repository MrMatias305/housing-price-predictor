import json
import os.path
import joblib
import numpy as np
from sklearn.model_selection import cross_val_score

from train_model import split_data


def validate_model():
    # Load model
    model_path = os.path.join("..", "models", "best_model_pipeline.pkl")
    with open(model_path, "rb") as f:
        model = joblib.load(model_path)

    # Cross validation
    X_train, X_test, y_train, y_test = split_data()
    mse_scores = cross_val_score(model, X_train, y_train,
                                 scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    r2_scores = cross_val_score(model, X_train, y_train,
                                scoring='r2', cv=5, n_jobs=-1)

    # Convert MSE values to RMSE
    rmse_scores = np.sqrt(-mse_scores)

    validation_metrics = {"mse_scores": (-mse_scores).tolist(),
                          "rmse_scores": rmse_scores.tolist(),
                          "r2_scores": r2_scores.tolist(),
                          "mean_mse": (-mse_scores).mean(),
                          "mean_rmse": rmse_scores.mean(),
                          "mean_r2": r2_scores.mean()
                          }

    # Save validation results
    model_path = os.path.join('..', 'results', 'reports', 'validation_metrics.json')
    with open(model_path, 'w') as f:
        json.dump(validation_metrics, f)


