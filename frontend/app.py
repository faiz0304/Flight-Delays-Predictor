"""
Streamlit Frontend for Flight Delay Prediction
Interactive UI for predicting flight delays using the backend API
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("✈️ Flight Delay Prediction Platform")
st.markdown("""
This application predicts flight delays using machine learning models based on historical aviation data.
Enter flight details to get a prediction for your flight.
""")

# Function to call the API
def predict_delay(flight_data):
    """Call the backend API to get delay prediction"""
    try:
        # For local development, we'll use the local API
        api_url = "http://localhost:8000/predict"
        response = requests.post(api_url, json=flight_data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend API. Please ensure the FastAPI server is running on localhost:8000")
        return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None

# Sidebar for flight inputs
st.sidebar.header("📝 Flight Details")

# Airline selector
airlines = {
    "American Airlines": "AA",
    "United Airlines": "UA",
    "Delta Air Lines": "DL",
    "Southwest Airlines": "SW",
    "JetBlue Airways": "B6",
    "Alaska Airlines": "AS",
    "Spirit Airlines": "NK",
    "Frontier Airlines": "F9",
    "Hawaiian Airlines": "HA"
}
airline_name = st.sidebar.selectbox("Airline", list(airlines.keys()))
airline_code = airlines[airline_name]

# Origin and destination airports
airports = {
    "Atlanta (ATL)": "ATL",
    "Los Angeles (LAX)": "LAX",
    "Chicago O'Hare (ORD)": "ORD",
    "Dallas/Fort Worth (DFW)": "DFW",
    "Denver (DEN)": "DEN",
    "New York JFK (JFK)": "JFK",
    "San Francisco (SFO)": "SFO",
    "Seattle (SEA)": "SEA",
    "Las Vegas (LAS)": "LAS",
    "Orlando (MCO)": "MCO"
}

origin_name = st.sidebar.selectbox("Origin Airport", list(airports.keys()))
origin_code = airports[origin_name]

destination_name = st.sidebar.selectbox("Destination Airport", list(airports.keys()))
destination_code = airports[destination_name]

# Date selection
flight_date = st.sidebar.date_input("Flight Date", value=datetime.now().date())

# Extract month and day of week from selected date
month = flight_date.month
day_of_week = flight_date.isoweekday()  # Monday is 1, Sunday is 7

# Flight distance (in miles)
distance = st.sidebar.slider("Flight Distance (miles)", min_value=100, max_value=3000, value=1500, step=50)

# Departure time
scheduled_departure_hour = st.sidebar.slider("Scheduled Departure Hour", min_value=0, max_value=23, value=14, step=1)

# Weather conditions
weather_conditions = ["Clear", "Cloudy", "Rain", "Snow", "Storm"]
weather_condition = st.sidebar.selectbox("Weather Condition", weather_conditions)

# Temperature
temperature = st.sidebar.slider("Temperature (°C)", min_value=-20, max_value=40, value=20, step=1)

# Humidity
humidity = st.sidebar.slider("Humidity (%)", min_value=0, max_value=100, value=60, step=5)

# Create flight data dictionary
flight_data = {
    "airline": airline_code,
    "origin": origin_code,
    "destination": destination_code,
    "month": month,
    "day_of_week": day_of_week,
    "distance": distance,
    "scheduled_departure_hour": scheduled_departure_hour,
    "temperature": temperature,
    "humidity": humidity,
    "weather_condition": weather_condition
}

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Flight Information")

    # Display flight details in a table format
    flight_info = {
        "Parameter": [
            "Airline",
            "Origin",
            "Destination",
            "Date",
            "Distance (miles)",
            "Departure Hour",
            "Weather",
            "Temperature (°C)",
            "Humidity (%)"
        ],
        "Value": [
            airline_name,
            origin_name,
            destination_name,
            flight_date.strftime("%Y-%m-%d"),
            f"{distance:,}",
            f"{scheduled_departure_hour}:00",
            weather_condition,
            temperature,
            humidity
        ]
    }

    flight_df = pd.DataFrame(flight_info)
    st.table(flight_df)

# Prediction button
with col2:
    st.subheader("Prediction")
    if st.button("Predict Flight Delay", type="primary"):
        with st.spinner("Analyzing flight data..."):
            # Get prediction from API
            prediction_result = predict_delay(flight_data)

            if prediction_result:
                delay_probability = prediction_result["delay_probability"]
                prediction = prediction_result["prediction"]
                delay_minutes = prediction_result["delay_minutes"]
                details = prediction_result["details"]

                # Display prediction results
                st.success(f"Prediction: **{prediction}**")

                # Display delay probability with gauge
                st.metric(
                    label="Delay Probability",
                    value=f"{delay_probability:.1%}",
                    delta="High risk" if delay_probability > 0.7 else "Medium risk" if delay_probability > 0.4 else "Low risk"
                )

                # Show predicted delay minutes
                if prediction == "Delayed":
                    st.metric(
                        label="Predicted Delay",
                        value=f"{delay_minutes:.1f} minutes"
                    )
                else:
                    st.metric(
                        label="Predicted Delay",
                        value="On time"
                    )

                # Show risk level
                risk_level = details["risk_level"]
                if risk_level == "High":
                    st.error(f"🚨 High Risk: {risk_level} chance of delay")
                elif risk_level == "Medium":
                    st.warning(f"⚠️ Medium Risk: {risk_level} chance of delay")
                else:
                    st.info(f"✅ Low Risk: {risk_level} chance of delay")

                # Additional details
                with st.expander("View Details"):
                    st.json(prediction_result)

# Visualization section
st.subheader("Delay Analysis")

# Create a mock analysis based on the input parameters
if st.button("Analyze Risk Factors"):
    # Create a bar chart showing how different factors might affect delay probability
    factors = ["Weather", "Time of Day", "Distance", "Day of Week", "Airline"]
    values = [
        0.3 if weather_condition in ["Rain", "Snow", "Storm"] else 0.1,
        0.25 if scheduled_departure_hour in [6, 7, 8, 16, 17, 18] else 0.1,  # Peak hours
        0.1 if distance > 1500 else 0.05,  # Long distance
        0.1 if day_of_week in [5, 6] else 0.05,  # Weekend
        0.1  # Airline factor (simplified)
    ]

    factor_df = pd.DataFrame({
        "Factor": factors,
        "Impact": values
    })

    fig = px.bar(
        factor_df,
        x="Impact",
        y="Factor",
        orientation="h",
        title="Risk Factor Impact on Delay Probability",
        color="Impact",
        color_continuous_scale="RdYlGn_r"  # Red to Green reversed (higher values are worse)
    )

    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <small>
        Flight Delay Prediction Platform | Based on historical aviation data and ML models<br>
        Note: This is a demo application. Actual flight delays may vary.
    </small>
</div>
""", unsafe_allow_html=True)