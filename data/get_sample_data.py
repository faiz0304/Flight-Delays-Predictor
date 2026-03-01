"""
Script to generate synthetic flight dataset matching the BTS schema for flight delay prediction.
Generates 1,000 rows of realistic flight data for model development and testing.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_synthetic_flight_data(n_samples=1000):
    """
    Generate synthetic flight data that mimics the BTS (Bureau of Transportation Statistics) schema.

    Features include:
    - Flight identifiers (airline, flight number)
    - Time-related features (departure/arrival times, day of week, etc.)
    - Airport information (origin/destination codes)
    - Aircraft and route information
    - Weather conditions (simulated)
    - Target variable: is_delayed (1 if arrival delay > 15 minutes, 0 otherwise)
    """

    # Create directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    print(f"Generating {n_samples} synthetic flight records...")

    # Define realistic values for categorical features
    airlines = ['AA', 'UA', 'DL', 'WN', 'US', 'B6', 'AS', 'NK', 'F9', 'HA']
    origin_airports = ['ATL', 'DFW', 'DEN', 'ORD', 'LAX', 'CLT', 'MCO', 'LAS', 'PHX', 'MIA',
                      'SEA', 'JFK', 'EWR', 'SFO', 'BOS', 'LGA', 'FLL', 'MSP', 'DTW', 'PHL']
    dest_airports = origin_airports.copy()

    # Generate synthetic data
    data = {
        'airline': [random.choice(airlines) for _ in range(n_samples)],
        'flight_num': [random.randint(1, 9999) for _ in range(n_samples)],
        'origin_airport': [random.choice(origin_airports) for _ in range(n_samples)],
        'dest_airport': [random.choice(dest_airports) for _ in range(n_samples)],
        'month': [random.randint(1, 12) for _ in range(n_samples)],
        'day_of_month': [random.randint(1, 28) for _ in range(n_samples)],  # Avoid day 29-31 issues
        'day_of_week': [random.randint(1, 7) for _ in range(n_samples)],
        'dep_time': [random.randint(0, 2359) for _ in range(n_samples)],  # 24-hour format
        'arr_time': [],  # Will be calculated based on dep_time + flight duration
        'dep_delay': [],  # Delay in departure
        'arr_delay': [],  # Delay in arrival (our target)
        'distance': [random.randint(100, 3000) for _ in range(n_samples)],  # Distance in miles
        'air_time': [],  # Flight time in minutes
        'carrier_delay': [],  # Delay due to carrier
        'weather_delay': [],  # Delay due to weather
        'nas_delay': [],  # Delay due to NAS (National Air System)
        'security_delay': [],  # Delay due to security
        'late_aircraft_delay': [],  # Delay due to late aircraft
        'is_delayed': []  # Target variable (1 if arrival delay > 15 min)
    }

    # Calculate dependent variables
    for i in range(n_samples):
        # Simulate air time based on distance (roughly 10 miles per minute for average)
        air_time = max(30, int(data['distance'][i] / 10 + random.gauss(0, 20)))
        data['air_time'].append(air_time)

        # Calculate arrival time based on departure time and air time
        dep_time = data['dep_time'][i]
        dep_hour = dep_time // 100
        dep_min = dep_time % 100
        # Add air time and convert to minutes
        total_dep_minutes = dep_hour * 60 + dep_min + air_time
        arr_hour = (total_dep_minutes // 60) % 24
        arr_min = total_dep_minutes % 60
        arr_time = arr_hour * 100 + arr_min
        data['arr_time'].append(arr_time)

        # Generate realistic delays
        # Departure delay: typically 0-60 minutes with some outliers
        dep_delay = max(0, random.gauss(10, 15))
        data['dep_delay'].append(dep_delay)

        # Factors affecting arrival delay
        carrier_factor = max(0, random.gauss(5, 10) if random.random() < 0.1 else 0)
        weather_factor = max(0, random.gauss(10, 20) if random.random() < 0.05 else 0)
        nas_factor = max(0, random.gauss(5, 8) if random.random() < 0.08 else 0)
        security_factor = max(0, random.gauss(15, 25) if random.random() < 0.02 else 0)
        late_aircraft_factor = max(0, random.gauss(8, 12) if random.random() < 0.07 else 0)

        # Total arrival delay is a combination of departure delay and other factors
        arr_delay = dep_delay + carrier_factor + weather_factor + nas_factor + security_factor + late_aircraft_factor
        data['arr_delay'].append(arr_delay)

        # Specific delay types
        data['carrier_delay'].append(carrier_factor)
        data['weather_delay'].append(weather_factor)
        data['nas_delay'].append(nas_factor)
        data['security_delay'].append(security_factor)
        data['late_aircraft_delay'].append(late_aircraft_factor)

        # Target variable: is_delayed = 1 if arrival delay > 15 minutes, else 0
        is_delayed = 1 if arr_delay > 15 else 0
        data['is_delayed'].append(is_delayed)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    output_path = os.path.join('data', 'synthetic_flight_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Synthetic flight data saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Delay rate: {df['is_delayed'].mean():.2%}")

    return df

if __name__ == "__main__":
    df = generate_synthetic_flight_data(n_samples=1000)
    print("\nFirst few rows of generated data:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nSample statistics:")
    print(df.describe())