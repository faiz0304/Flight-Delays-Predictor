"""
ETL preprocessing script for flight delay prediction data.
This script handles data cleaning, feature engineering, and preparation for ML modeling.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import pickle

def load_data(file_path):
    """Load flight data from CSV"""
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records")
    return df

def clean_data(df):
    """Clean the flight data"""
    print("Starting data cleaning...")

    initial_count = len(df)

    # Remove rows with missing critical values
    df = df.dropna(subset=['arr_delay', 'dep_delay', 'distance'])

    # Remove unrealistic values
    df = df[df['distance'] > 0]  # Distance must be positive
    df = df[df['arr_delay'] >= -50]  # Arrival delay shouldn't be extremely negative
    df = df[df['arr_delay'] < 1000]  # Cap extremely large delays
    df = df[df['dep_delay'] >= -50]  # Departure delay shouldn't be extremely negative
    df = df[df['dep_delay'] < 1000]  # Cap extremely large delays

    # Remove flights with 0 air time
    df = df[df['air_time'] > 0]

    print(f"Removed {initial_count - len(df)} records during cleaning")
    print(f"Final count: {len(df)} records")

    return df

def engineer_features(df):
    """Create new features from existing data"""
    print("Starting feature engineering...")

    # Create time-based features
    df['dep_hour'] = df['dep_time'] // 100
    df['arr_hour'] = df['arr_time'] // 100

    # Create distance-based categories
    df['distance_category'] = pd.cut(df['distance'],
                                    bins=[0, 500, 1000, 2000, float('inf')],
                                    labels=['Short', 'Medium', 'Long', 'Very Long'])

    # Create time-based categories
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [6, 7] else 0)
    df['is_holiday_season'] = df['month'].apply(lambda x: 1 if x in [7, 8, 11, 12] else 0)

    # Create airport traffic indicators (simplified - in real scenario, this would be from external data)
    busy_airports = ['ATL', 'ORD', 'LAX', 'DFW', 'JFK', 'SFO']
    df['origin_is_busy'] = df['origin_airport'].apply(lambda x: 1 if x in busy_airports else 0)
    df['dest_is_busy'] = df['dest_airport'].apply(lambda x: 1 if x in busy_airports else 0)

    # Calculate speed (simplified)
    df['avg_speed'] = df['distance'] / (df['air_time'] / 60.0)

    # Encode categorical variables
    le_airline = LabelEncoder()
    le_origin = LabelEncoder()
    le_dest = LabelEncoder()
    le_distance_cat = LabelEncoder()

    df['airline_encoded'] = le_airline.fit_transform(df['airline'])
    df['origin_airport_encoded'] = le_origin.fit_transform(df['origin_airport'])
    df['dest_airport_encoded'] = le_dest.fit_transform(df['dest_airport'])
    df['distance_category_encoded'] = le_distance_cat.fit_transform(df['distance_category'].astype(str))

    # Store encoders for later use in inference
    encoders = {
        'airline': le_airline,
        'origin_airport': le_origin,
        'dest_airport': le_dest,
        'distance_category': le_distance_cat
    }

    return df, encoders

def prepare_features(df, target_column='is_delayed'):
    """Prepare feature matrix X and target vector y"""
    print("Preparing features for modeling...")

    # Select features for the model
    feature_columns = [
        'airline_encoded',
        'origin_airport_encoded',
        'dest_airport_encoded',
        'month',
        'day_of_month',
        'day_of_week',
        'dep_hour',
        'arr_hour',
        'dep_delay',
        'distance',
        'air_time',
        'distance_category_encoded',
        'is_weekend',
        'is_holiday_season',
        'origin_is_busy',
        'dest_is_busy',
        'avg_speed',
        'carrier_delay',
        'weather_delay',
        'nas_delay',
        'security_delay',
        'late_aircraft_delay'
    ]

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    return X, y

def preprocess_pipeline(file_path):
    """Full preprocessing pipeline"""
    print("Starting preprocessing pipeline...")

    # Load data
    df = load_data(file_path)

    # Clean data
    df = clean_data(df)

    # Engineer features
    df, encoders = engineer_features(df)

    # Prepare features and target
    X, y = prepare_features(df)

    print("Preprocessing pipeline completed successfully!")
    return X, y, encoders

def save_encoders(encoders, file_path):
    """Save encoders to file"""
    with open(file_path, 'wb') as f:
        pickle.dump(encoders, f)
    print(f"Encoders saved to {file_path}")

def main():
    """Main preprocessing function"""
    # Define file paths
    input_file = os.path.join('data', 'synthetic_flight_data.csv')
    encoder_file = os.path.join('models', 'encoders.pkl')

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Run preprocessing pipeline
    X, y, encoders = preprocess_pipeline(input_file)

    # Save encoders
    save_encoders(encoders, encoder_file)

    # Save processed data
    processed_data_path = os.path.join('data', 'processed_flight_data.csv')
    df_combined = X.copy()
    df_combined['is_delayed'] = y
    df_combined.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")

    # Print summary
    print(f"\nFinal dataset summary:")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Delay rate: {y.mean():.2%}")

    return X, y, encoders

if __name__ == "__main__":
    X, y, encoders = main()