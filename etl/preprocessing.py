"""
ETL Pipeline for Flight Data
This module handles the extraction, transformation, and loading of flight data.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_flight_data(filepath):
    """
    Load flight data from CSV
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} records with columns: {list(df.columns)}")
    return df

def clean_data(df):
    """
    Clean flight data by handling missing values and filtering outliers
    """
    logger.info("Starting data cleaning...")

    initial_count = len(df)

    # Remove records with missing critical values
    df = df.dropna(subset=['arrival_delay', 'airline', 'origin', 'destination'])

    # Filter out extreme delay values (likely data errors)
    df = df[(df['arrival_delay'] >= 0) & (df['arrival_delay'] <= 1200)]  # Max 20 hours

    # Remove duplicate flights (same date, airline, origin, destination)
    df = df.drop_duplicates(subset=['flight_date', 'airline', 'origin', 'destination'], keep='first')

    logger.info(f"Cleaned data: {initial_count} -> {len(df)} records ({initial_count - len(df)} removed)")
    return df

def engineer_features(df):
    """
    Engineer features for the flight delay prediction model
    """
    logger.info("Starting feature engineering...")

    # Create route identifier
    df['route'] = df['origin'] + '_' + df['destination']

    # Create distance categories
    df['distance_category'] = pd.cut(df['distance'], bins=[0, 500, 1000, 1500, np.inf],
                                    labels=['Short', 'Medium', 'Long', 'Very Long'])

    # Create time-based features
    df['is_weekend'] = (df['day_of_week'] >= 6).astype(int)
    df['is_holiday_season'] = ((df['month'] == 12) | (df['month'] == 1)).astype(int)
    df['is_peak_travel'] = df['scheduled_departure_hour'].apply(
        lambda x: 1 if x in [6, 7, 8, 16, 17, 18] else 0
    )

    # Weather impact feature
    weather_impact_map = {'Clear': 0, 'Cloudy': 1, 'Rain': 2, 'Snow': 3, 'Storm': 4}
    df['weather_impact'] = df['weather_condition'].map(weather_impact_map)

    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['airline', 'origin', 'destination', 'route', 'distance_category']

    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            # Fit on all possible values to avoid unseen labels during prediction
            le.fit(df[col].astype(str))
            df[f'{col}_encoded'] = le.transform(df[col].astype(str))
            label_encoders[col] = le

    logger.info("Feature engineering completed")
    logger.info(f"Engineered features: {len([col for col in df.columns if col.endswith('_encoded')])} encoded features added")

    return df, label_encoders

def prepare_features_target(df):
    """
    Prepare feature matrix X and target vector y
    """
    logger.info("Preparing features and target...")

    # Define feature columns for the model
    feature_columns = [
        'airline_encoded', 'origin_encoded', 'destination_encoded', 'route_encoded',
        'month', 'day_of_week', 'distance', 'distance_category_encoded',
        'scheduled_departure_hour', 'temperature', 'humidity', 'weather_impact',
        'is_weekend', 'is_holiday_season', 'is_peak_travel'
    ]

    # Ensure all required columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    X = df[feature_columns].copy()
    y_classification = df['is_delayed'].values
    y_regression = df['arrival_delay'].values  # For regression task

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Classification target shape: {y_classification.shape}")
    logger.info(f"Regression target shape: {y_regression.shape}")

    return X, y_classification, y_regression

def preprocess_data(input_filepath, output_dir):
    """
    Complete ETL pipeline: Load, clean, engineer features, and save processed data
    """
    logger.info("Starting ETL pipeline...")

    # Load data
    df = load_flight_data(input_filepath)

    # Clean data
    df = clean_data(df)

    # Engineer features
    df, label_encoders = engineer_features(df)

    # Prepare features and targets
    X, y_class, y_reg = prepare_features_target(df)

    # Save processed data
    os.makedirs(output_dir, exist_ok=True)

    X_output_path = os.path.join(output_dir, 'X_processed.csv')
    y_class_output_path = os.path.join(output_dir, 'y_classification.csv')
    y_reg_output_path = os.path.join(output_dir, 'y_regression.csv')

    X.to_csv(X_output_path, index=False)
    pd.DataFrame(y_class, columns=['is_delayed']).to_csv(y_class_output_path, index=False)
    pd.DataFrame(y_reg, columns=['arrival_delay']).to_csv(y_reg_output_path, index=False)

    # Save label encoders as a dictionary
    import joblib
    encoders_path = os.path.join(output_dir, 'label_encoders.pkl')
    joblib.dump(label_encoders, encoders_path)

    logger.info(f"Processed data saved:")
    logger.info(f"  Features: {X_output_path}")
    logger.info(f"  Classification targets: {y_class_output_path}")
    logger.info(f"  Regression targets: {y_reg_output_path}")
    logger.info(f"  Label encoders: {encoders_path}")

    # Print summary
    logger.info(f"Final dataset shape: {X.shape}")
    logger.info(f"Delayed flights: {sum(y_class)} ({sum(y_class)/len(y_class):.2%})")

    return X, y_class, y_reg, label_encoders

if __name__ == "__main__":
    # Define file paths
    raw_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw_flight_data.csv')
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')

    # Run the preprocessing pipeline
    try:
        X, y_class, y_reg, encoders = preprocess_data(raw_data_path, output_dir)
        print("ETL pipeline completed successfully!")
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_data_path}")
        print("Please run 'python data/generate_synthetic_data.py' first")
    except Exception as e:
        logger.error(f"Error in ETL pipeline: {str(e)}")
        raise