
# for data manipulation
import pandas as pd
import numpy as np

# for preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    average_precision_score
)

# for model serialization
import os
import joblib
import mlflow

# MLflow configuration
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(
    "Wellness_Tourism_Purchase_Prediction"
)

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact

split_dir = "tourism_project/model_building/data_splits"

Xtrain = pd.read_csv(os.path.join(split_dir, "Xtrain.csv"))
Xtest = pd.read_csv(os.path.join(split_dir, "Xtest.csv"))
ytrain = pd.read_csv(os.path.join(split_dir, "ytrain.csv")).squeeze()
ytest = pd.read_csv(os.path.join(split_dir, "ytest.csv")).squeeze()

# Identify numerical and categorical features

numeric_features = Xtrain.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = Xtrain.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# Set the class weight to handle class imbalance

class_weight = (
    (ytrain == 0).sum() /
    (ytrain == 1).sum()
)

print("Class weight:", class_weight)

# Define the preprocessing steps

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Define base XGBoost model

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss",
    n_jobs=1
)

# Hyperparameter grid based on the development experiment

param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.8, 1.0]
}

# Model pipeline

model_pipeline = make_pipeline(preprocessor, xgb_model)

# Classification threshold selected for the current experiment

classification_threshold = 0.45

# Start MLflow run

with mlflow.start_run():

    # Hyperparameter tuning

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=cv,
        scoring="average_precision",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores

    results = grid_search.cv_results_

    for i in range(len(results["params"])):

        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        # Log each combination as a separate MLflow run

        with mlflow.start_run(nested=True):

            mlflow.log_params(param_set)

            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run

    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model

    best_model = grid_search.best_estimator_

    print("\nBest params:", grid_search.best_params_)

    print("\nBest Cross Validation PR-AUC:")
    print(grid_search.best_score_)

    mlflow.log_metric(
        "best_cv_pr_auc",
        grid_search.best_score_
    )

    # ======================================================
    # PREDICTION AND MODEL EVALUATION
    # ======================================================

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]

    y_pred_train = (
        y_pred_train_proba >= classification_threshold
    ).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]

    y_pred_test = (
        y_pred_test_proba >= classification_threshold
    ).astype(int)

    train_report = classification_report(
        ytrain,
        y_pred_train,
        output_dict=True,
        zero_division=0
    )

    test_report = classification_report(
        ytest,
        y_pred_test,
        output_dict=True,
        zero_division=0
    )

    print("\nTraining Classification Report:")
    print(classification_report(ytrain, y_pred_train))

    print("\nTesting Classification Report:")
    print(classification_report(ytest, y_pred_test))

    # Log the metrics for the best model

    mlflow.log_metrics({

        "classification_threshold": classification_threshold,

        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1_score": train_report["1"]["f1-score"],

        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1_score": test_report["1"]["f1-score"],

        "test_roc_auc": roc_auc_score(
            ytest, y_pred_test_proba
        ),

        "test_pr_auc": average_precision_score(
            ytest, y_pred_test_proba
        )
    })

    # Save next to app.py so the Streamlit app can load it directly,
    # and log it as an MLflow artifact for traceability.

    model_dir = "tourism_project/deployment"

    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(
        model_dir,
        "best_tourism_model.joblib"
    )

    model_bundle = {
        "model": best_model,
        "classification_threshold": classification_threshold
    }

    joblib.dump(model_bundle, model_path)

    mlflow.log_artifact(model_path, artifact_path="model")

    print(f"\nModel saved to {model_path}")

    print("\nAll XGBoost experiments logged successfully.")
