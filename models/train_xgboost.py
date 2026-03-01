"""
Flight Delay Prediction Model Training
Trains XGBoost models for both classification and regression tasks
"""
import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import mlflow
import mlflow.xgboost
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Flight_Delay_Prediction")

def load_processed_data(data_dir):
    """
    Load processed features and targets
    """
    logger.info(f"Loading processed data from {data_dir}")

    X = pd.read_csv(os.path.join(data_dir, 'X_processed.csv'))
    y_class = pd.read_csv(os.path.join(data_dir, 'y_classification.csv')).values.ravel()
    y_reg = pd.read_csv(os.path.join(data_dir, 'y_regression.csv')).values.ravel()

    logger.info(f"Loaded feature matrix: {X.shape}")
    logger.info(f"Loaded classification targets: {y_class.shape}")
    logger.info(f"Loaded regression targets: {y_reg.shape}")

    return X, y_class, y_reg

def train_classification_model(X, y):
    """
    Train XGBoost model for classification (delayed/not delayed)
    """
    logger.info("Starting classification model training...")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define the model with optimized parameters
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    # Train the model
    with mlflow.start_run(run_name=f"classification_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        logger.info("Fitting classification model...")
        model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        auc = roc_auc_score(y_test, y_test_proba)

        logger.info(f"Classification Model Metrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  ROC-AUC: {auc:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # Log model parameters
        mlflow.log_params(model.get_params())

        # Log the model to MLflow
        mlflow.xgboost.log_model(model, "classification_model")

        # Log feature importance as an artifact
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        importance_path = "feature_importance_classification.csv"
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        # Log the test dataset for model validation
        X_test_path = "X_test_classification.csv"
        y_test_path = "y_test_classification.csv"
        pd.DataFrame(X_test).to_csv(X_test_path, index=False)
        pd.DataFrame(y_test).to_csv(y_test_path, index=False)
        mlflow.log_artifact(X_test_path)
        mlflow.log_artifact(y_test_path)

        # Save model locally as well
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'classification_model.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

        logger.info(f"Classification model saved to {model_path}")
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    return model, X_test, y_test, y_test_pred, y_test_proba

def train_regression_model(X, y):
    """
    Train XGBoost model for regression (predict delay minutes)
    """
    logger.info("Starting regression model training...")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model with optimized parameters
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    # Train the model
    with mlflow.start_run(run_name=f"regression_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        logger.info("Fitting regression model...")
        model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_test_pred)
        r2 = model.score(X_test, y_test)

        logger.info(f"Regression Model Metrics:")
        logger.info(f"  MSE: {mse:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R²: {r2:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        # Log model parameters
        mlflow.log_params(model.get_params())

        # Log the model to MLflow
        mlflow.xgboost.log_model(model, "regression_model")

        # Log feature importance as an artifact
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        importance_path = "feature_importance_regression.csv"
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        # Log the test dataset for model validation
        X_test_path = "X_test_regression.csv"
        y_test_path = "y_test_regression.csv"
        pd.DataFrame(X_test).to_csv(X_test_path, index=False)
        pd.DataFrame(y_test).to_csv(y_test_path, index=False)
        mlflow.log_artifact(X_test_path)
        mlflow.log_artifact(y_test_path)

        # Save model locally as well
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'regression_model.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

        logger.info(f"Regression model saved to {model_path}")
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    return model, X_test, y_test, y_test_pred

def evaluate_models(class_model, reg_model, X_test_class, y_test_class, y_test_class_pred, y_test_class_proba, X_test_reg, y_test_reg, y_test_reg_pred):
    """
    Evaluate both models and print comprehensive results
    """
    logger.info("Evaluating models...")

    # Classification metrics
    class_accuracy = accuracy_score(y_test_class, y_test_class_pred)
    class_precision = precision_score(y_test_class, y_test_class_pred)
    class_recall = recall_score(y_test_class, y_test_class_pred)
    class_f1 = f1_score(y_test_class, y_test_class_pred)
    class_auc = roc_auc_score(y_test_class, y_test_class_proba)

    # Regression metrics
    reg_mse = mean_squared_error(y_test_reg, y_test_reg_pred)
    reg_rmse = np.sqrt(reg_mse)
    reg_mae = mean_absolute_error(y_test_reg, y_test_reg_pred)
    reg_r2 = reg_model.score(X_test_reg, y_test_reg)

    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)

    print("\nClassification Model (Delay Prediction >15min):")
    print(f"  Accuracy:  {class_accuracy:.4f}")
    print(f"  Precision: {class_precision:.4f}")
    print(f"  Recall:    {class_recall:.4f}")
    print(f"  F1-Score:  {class_f1:.4f}")
    print(f"  ROC-AUC:   {class_auc:.4f}")

    print("\nRegression Model (Delay Minutes Prediction):")
    print(f"  MSE:   {reg_mse:.4f}")
    print(f"  RMSE:  {reg_rmse:.4f}")
    print(f"  MAE:   {reg_mae:.4f}")
    print(f"  R²:    {reg_r2:.4f}")

    print("\n" + "="*60)

    # Check if we met our performance goals
    if class_auc > 0.82:
        print("SUCCESS: Classification model met ROC-AUC target (>0.82)!")
    else:
        print("WARNING: Classification model did not meet ROC-AUC target (>0.82)")

    if reg_rmse < 20:  # Assuming <20 min error is acceptable
        print("SUCCESS: Regression model met performance target!")
    else:
        print("WARNING: Regression model may need improvement")

    print("="*60)

def main():
    """
    Main training pipeline
    """
    logger.info("Starting flight delay prediction model training...")

    # Define paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')

    try:
        # Load processed data
        X, y_class, y_reg = load_processed_data(data_dir)

        # Train classification model
        class_model, X_test_class, y_test_class, y_test_class_pred, y_test_class_proba = train_classification_model(X, y_class)

        # Train regression model
        reg_model, X_test_reg, y_test_reg, y_test_reg_pred = train_regression_model(X, y_reg)

        # Evaluate models
        evaluate_models(
            class_model, reg_model,
            X_test_class, y_test_class, y_test_class_pred, y_test_class_proba,
            X_test_reg, y_test_reg, y_test_reg_pred
        )

        logger.info("Model training completed successfully!")
        logger.info("Models saved to the models/ directory")
        logger.info("MLflow tracking enabled - check the MLflow UI for detailed experiment tracking")

    except FileNotFoundError:
        logger.error(f"Processed data not found in {data_dir}")
        logger.error("Please run 'python etl/preprocessing.py' first to process the raw data")
        raise
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    main()