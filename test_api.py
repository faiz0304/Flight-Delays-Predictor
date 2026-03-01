"""
Test script to verify the Flight Delay Prediction API
"""
import requests
import json

def test_api():
    print("Testing Flight Delay Prediction API...")

    # Test data
    test_flight = {
        "airline": "AA",
        "origin": "ATL",
        "destination": "LAX",
        "month": 3,
        "day_of_week": 3,
        "distance": 2000,
        "scheduled_departure_hour": 14,
        "temperature": 22.5,
        "humidity": 65.0,
        "weather_condition": "Clear"
    }

    try:
        # Make request to the API
        response = requests.post("http://localhost:8000/predict", json=test_flight)

        if response.status_code == 200:
            result = response.json()
            print("✅ API Test Successful!")
            print(f"Delay Probability: {result['delay_probability']:.2%}")
            print(f"Prediction: {result['prediction']}")
            print(f"Delay Minutes: {result['delay_minutes']:.2f}")
            print(f"Risk Level: {result['details']['risk_level']}")
        else:
            print(f"❌ API Test Failed with status code: {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the backend is running on localhost:8000")
    except Exception as e:
        print(f"❌ Error during API test: {str(e)}")

if __name__ == "__main__":
    test_api()