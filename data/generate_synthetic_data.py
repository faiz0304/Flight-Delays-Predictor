"""
Synthetic Flight Data Generator
Generates realistic flight data for the flight delay prediction project.
"""
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_synthetic_flight_data(n_samples=10000):
    """
    Generate synthetic flight data with realistic features
    """
    print("Generating synthetic flight data...")

    # Define realistic flight data categories
    airlines = ['AA', 'UA', 'DL', 'SW', 'WN', 'B6', 'AS', 'NK', 'F9', 'HA']
    origins = ['ATL', 'LAX', 'ORD', 'DFW', 'DEN', 'JFK', 'SFO', 'SEA', 'LAS', 'MCO']
    destinations = ['ATL', 'LAX', 'ORD', 'DFW', 'DEN', 'JFK', 'SFO', 'SEA', 'LAS', 'MCO']

    # Weather conditions (affects delay probability)
    weather_conditions = ['Clear', 'Cloudy', 'Rain', 'Snow', 'Storm']
    weather_impact = {'Clear': 0.05, 'Cloudy': 0.08, 'Rain': 0.15, 'Snow': 0.25, 'Storm': 0.4}

    data = {
        'airline': [random.choice(airlines) for _ in range(n_samples)],
        'origin': [random.choice(origins) for _ in range(n_samples)],
        'destination': [random.choice(destinations) for _ in range(n_samples)],
        'month': [random.randint(1, 12) for _ in range(n_samples)],
        'day_of_week': [random.randint(1, 7) for _ in range(n_samples)],
        'distance': [random.randint(100, 2500) for _ in range(n_samples)],
        'scheduled_departure_hour': [random.randint(0, 23) for _ in range(n_samples)],
        'temperature': [random.uniform(-10, 40) for _ in range(n_samples)],  # Celsius
        'humidity': [random.uniform(20, 90) for _ in range(n_samples)],
        'weather_condition': [random.choice(weather_conditions) for _ in range(n_samples)],
    }

    df = pd.DataFrame(data)

    # Calculate delay based on realistic factors
    delay_minutes = []
    for i in range(n_samples):
        base_delay = 0

        # Base delay influenced by various factors
        if df.iloc[i]['day_of_week'] in [5, 6]:  # Friday, Saturday
            base_delay += 5  # Weekend flights often delayed more
        if df.iloc[i]['scheduled_departure_hour'] in [6, 7, 8, 16, 17, 18]:  # Rush hours
            base_delay += 8  # Peak hours have more delays
        if df.iloc[i]['month'] in [11, 12, 1, 2]:  # Winter months
            base_delay += 10  # More delays in winter
        if df.iloc[i]['distance'] > 1500:  # Long distance flights
            base_delay += 3  # Longer flights have different delay patterns

        # Weather impact
        weather = df.iloc[i]['weather_condition']
        weather_factor = weather_impact[weather]
        delay_probability = base_delay * 0.01 + weather_factor

        # Add some randomness
        if random.random() < delay_probability:
            # Random delay between 0 and 240 minutes (4 hours)
            actual_delay = base_delay + random.gauss(0, 15) + random.uniform(0, 60)
            delay_minutes.append(max(0, actual_delay))
        else:
            delay_minutes.append(0)

    df['arrival_delay'] = delay_minutes

    # Create binary target: 1 if delay > 15 minutes, 0 otherwise
    df['is_delayed'] = (df['arrival_delay'] > 15).astype(int)

    # Create a datetime column for completeness
    base_date = datetime.now() - timedelta(days=365)
    df['flight_date'] = [base_date + timedelta(days=random.randint(0, 365)) for _ in range(n_samples)]

    print(f"Generated {len(df)} flight records")
    return df

def save_data(df, output_path):
    """
    Save the generated data to CSV
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    # Generate data
    df = generate_synthetic_flight_data(10000)

    # Save to data directory
    output_path = os.path.join(os.path.dirname(__file__), 'raw_flight_data.csv')
    save_data(df, output_path)

    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total records: {len(df)}")
    print(f"Delayed flights: {df['is_delayed'].sum()} ({df['is_delayed'].mean():.2%})")
    print(f"Average delay: {df['arrival_delay'].mean():.2f} minutes")
    print("\nColumn names:", list(df.columns))