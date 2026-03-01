"""
Model training script for flight delay prediction.
Uses XGBoost to train a binary classification model to predict flight delays > 15 minutes.
Logs metrics to MLflow for experiment tracking.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import mlflow
import mlflow.xgboost
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_processed_data():
    """Load the preprocessed flight data"""
    print("Loading processed flight data...")
    data_path = os.path.join('data', 'processed_flight_data.csv')
    df = pd.read_csv(data_path)

    # Separate features and target
    y = df['is_delayed']
    X = df.drop('is_delayed', axis=1)

    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Target distribution: {y.value_counts(normalize=True).to_dict()}")

    return X, y

def train_model():
    """Train the XGBoost model with MLflow tracking"""
    print("Starting model training...")

    # Set MLflow tracking URI to local SQLite
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Flight_Delay_Prediction")

    # Load data
    X, y = load_processed_data()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Initialize MLflow run
    with mlflow.start_run():
        # Define model parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }

        # Log parameters
        mlflow.log_params(params)

        # Train the model
        print("Training XGBoost model...")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_proba)

        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_auc", test_auc)

        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Test AUC: {test_auc:.4f}")

        # Log the model
        mlflow.xgboost.log_model(model, "model")

        # Save the model to models directory as well
        model_path = os.path.join('models', 'flight_model.joblib')
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred))

        # Feature importance
        feature_importance = model.feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))

        # Log feature importance as an artifact
        importance_path = os.path.join('models', 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        # Log training details
        mlflow.set_tag("model_type", "XGBoost Binary Classifier")
        mlflow.set_tag("target", "is_delayed (arrival_delay > 15 min)")

        print(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")
        print("Model training completed and logged to MLflow!")

        return model, X_test, y_test, y_test_pred, y_test_proba

def main():
    """Main training function"""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Train the model
    model, X_test, y_test, y_test_pred, y_test_proba = train_model()

    print("\nModel training pipeline completed successfully!")

    # Verify model file exists
    model_path = os.path.join('models', 'flight_model.joblib')
    if os.path.exists(model_path):
        print(f"✅ Model file exists at: {model_path}")
        print(f"Model file size: {os.path.getsize(model_path)} bytes")
    else:
        print("❌ Model file was not created!")

if __name__ == "__main__":
    main()