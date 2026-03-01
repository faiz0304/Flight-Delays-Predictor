"""
FastAPI Backend for Flight Delay Prediction
Provides endpoints for flight delay prediction using trained XGBoost models
"""
import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import logging
import datetime
from sklearn.preprocessing import LabelEncoder

# Pydantic models for request/response (defined here instead of separate file)
class FlightInput(BaseModel):
    airline: str = Field(..., example="AA", description="Airline code (e.g., AA, UA, DL)")
    origin: str = Field(..., example="ATL", description="Origin airport code")
    destination: str = Field(..., example="LAX", description="Destination airport code")
    month: int = Field(..., ge=1, le=12, example=3, description="Month of flight (1-12)")
    day_of_week: int = Field(..., ge=1, le=7, example=3, description="Day of week (1=Monday, 7=Sunday)")
    distance: float = Field(..., ge=0, example=2000, description="Flight distance in miles")
    scheduled_departure_hour: int = Field(..., ge=0, le=23, example=14, description="Scheduled departure hour (0-23)")
    temperature: float = Field(default=20.0, ge=-50, le=50, example=22.5, description="Temperature in Celsius")
    humidity: float = Field(default=60.0, ge=0, le=100, example=65.0, description="Humidity percentage")
    weather_condition: str = Field(default="Clear", example="Clear", description="Weather condition")

class PredictionResponse(BaseModel):
    delay_probability: float = Field(..., example=0.78, description="Probability of delay > 15 minutes")
    prediction: str = Field(..., example="Delayed", description="Prediction result (Delayed/On Time)")
    delay_minutes: float = Field(..., example=25.5, description="Predicted delay in minutes")
    details: Dict[str, Any] = Field(..., description="Additional prediction details")

class BatchPredictionRequest(BaseModel):
    flights: List[FlightInput]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Flight Delay Prediction API",
    description="API for predicting flight delays using machine learning models",
    version="1.0.0"
)

# Add CORS middleware to allow requests from web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class FlightInput(BaseModel):
    airline: str = Field(..., example="AA", description="Airline code (e.g., AA, UA, DL)")
    origin: str = Field(..., example="ATL", description="Origin airport code")
    destination: str = Field(..., example="LAX", description="Destination airport code")
    month: int = Field(..., ge=1, le=12, example=3, description="Month of flight (1-12)")
    day_of_week: int = Field(..., ge=1, le=7, example=3, description="Day of week (1=Monday, 7=Sunday)")
    distance: float = Field(..., ge=0, example=2000, description="Flight distance in miles")
    scheduled_departure_hour: int = Field(..., ge=0, le=23, example=14, description="Scheduled departure hour (0-23)")
    temperature: float = Field(default=20.0, example=22.5, description="Temperature in Celsius")
    humidity: float = Field(default=60.0, example=65.0, description="Humidity percentage")
    weather_condition: str = Field(default="Clear", example="Clear", description="Weather condition")

class PredictionResponse(BaseModel):
    delay_probability: float = Field(..., example=0.78, description="Probability of delay > 15 minutes")
    prediction: str = Field(..., example="Delayed", description="Prediction result (Delayed/On Time)")
    delay_minutes: float = Field(..., example=25.5, description="Predicted delay in minutes")
    details: Dict[str, Any] = Field(..., description="Additional prediction details")

class BatchPredictionRequest(BaseModel):
    flights: List[FlightInput]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

# Global variables to hold loaded models and encoders
classification_model = None
regression_model = None
label_encoders = None

def load_models():
    """Load trained models and label encoders at startup"""
    global classification_model, regression_model, label_encoders

    try:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

        logger.info("Loading classification model...")
        classification_model_path = os.path.join(model_dir, 'classification_model.pkl')
        classification_model = joblib.load(classification_model_path)
        logger.info(f"Classification model loaded from {classification_model_path}")

        logger.info("Loading regression model...")
        regression_model_path = os.path.join(model_dir, 'regression_model.pkl')
        regression_model = joblib.load(regression_model_path)
        logger.info(f"Regression model loaded from {regression_model_path}")

        logger.info("Loading label encoders...")
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
        encoders_path = os.path.join(data_dir, 'label_encoders.pkl')
        label_encoders = joblib.load(encoders_path)
        logger.info(f"Label encoders loaded from {encoders_path}")

    except FileNotFoundError as e:
        logger.error(f"Model file not found: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model file not found: {str(e)}")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features using saved label encoders
    """
    df = df.copy()

    categorical_columns = ['airline', 'origin', 'destination', 'distance_category']

    # Create route identifier
    df['route'] = df['origin'] + '_' + df['destination']
    categorical_columns.append('route')

    # Create distance categories
    df['distance_category'] = pd.cut(
        df['distance'],
        bins=[0, 500, 1000, 1500, float('inf')],
        labels=['Short', 'Medium', 'Long', 'Very Long']
    )

    # Apply encoding for each categorical column
    for col in categorical_columns:
        if col in df.columns:
            # Handle unknown categories by mapping them to the most frequent category
            if col in label_encoders:
                le = label_encoders[col]

                # For new/unseen categories, we'll map them to the first encoded value
                # In production, you might want to handle this differently
                df[f'{col}_encoded'] = df[col].apply(
                    lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else 0
                )
            else:
                # If encoder doesn't exist, create a simple mapping
                unique_vals = df[col].unique()
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                df[f'{col}_encoded'] = df[col].map(mapping).fillna(0).astype(int)

    # Additional feature engineering (same as in training)
    df['is_weekend'] = (df['day_of_week'] >= 6).astype(int)
    df['is_holiday_season'] = ((df['month'] == 12) | (df['month'] == 1)).astype(int)
    df['is_peak_travel'] = df['scheduled_departure_hour'].apply(
        lambda x: 1 if x in [6, 7, 8, 16, 17, 18] else 0
    )

    # Weather impact
    weather_impact_map = {'Clear': 0, 'Cloudy': 1, 'Rain': 2, 'Snow': 3, 'Storm': 4}
    df['weather_impact'] = df['weather_condition'].map(weather_impact_map).fillna(0).astype(int)

    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features in the same way as during training
    """
    # Encode categorical features
    df = encode_categorical_features(df)

    # Define feature columns (must match training)
    feature_columns = [
        'airline_encoded', 'origin_encoded', 'destination_encoded', 'route_encoded',
        'month', 'day_of_week', 'distance', 'distance_category_encoded',
        'scheduled_departure_hour', 'temperature', 'humidity', 'weather_impact',
        'is_weekend', 'is_holiday_season', 'is_peak_travel'
    ]

    # Ensure all required columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0  # Default value for missing columns

    return df[feature_columns]

@app.on_event("startup")
async def startup_event():
    """Load models when the application starts"""
    logger.info("Starting up Flight Delay Prediction API...")
    load_models()
    logger.info("API startup completed successfully!")

@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {
        "message": "Flight Delay Prediction API",
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": classification_model is not None and regression_model is not None,
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_delay(flight: FlightInput):
    """
    Predict flight delay for a single flight
    """
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([flight.dict()])

        # Prepare features
        X = prepare_features(input_data)

        # Make predictions
        delay_probability = classification_model.predict_proba(X)[:, 1][0]
        is_delayed = classification_model.predict(X)[0]
        delay_minutes = regression_model.predict(X)[0]

        # Format prediction result
        prediction = "Delayed" if is_delayed else "On Time"

        # Ensure delay minutes is non-negative
        delay_minutes = max(0, float(delay_minutes))

        response = PredictionResponse(
            delay_probability=float(delay_probability),
            prediction=prediction,
            delay_minutes=delay_minutes,
            details={
                "flight_info": flight.dict(),
                "confidence": float(delay_probability) if is_delayed else float(1 - delay_probability),
                "risk_level": "High" if delay_probability > 0.7 else "Medium" if delay_probability > 0.4 else "Low"
            }
        )

        logger.info(f"Prediction made - Probability: {delay_probability:.3f}, Delay: {delay_minutes:.2f}min")
        return response

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_delay_batch(request: BatchPredictionRequest):
    """
    Predict flight delays for multiple flights
    """
    try:
        # Convert inputs to DataFrame
        flights_data = [flight.dict() for flight in request.flights]
        df = pd.DataFrame(flights_data)

        # Prepare features
        X = prepare_features(df)

        # Make predictions
        delay_probabilities = classification_model.predict_proba(X)[:, 1]
        is_delayed_predictions = classification_model.predict(X)
        delay_minutes_predictions = regression_model.predict(X)

        # Format responses
        predictions = []
        for i in range(len(request.flights)):
            prediction = "Delayed" if is_delayed_predictions[i] else "On Time"
            delay_minutes = max(0, float(delay_minutes_predictions[i]))

            pred_response = PredictionResponse(
                delay_probability=float(delay_probabilities[i]),
                prediction=prediction,
                delay_minutes=delay_minutes,
                details={
                    "flight_info": request.flights[i].dict(),
                    "confidence": float(delay_probabilities[i]) if is_delayed_predictions[i] else float(1 - delay_probabilities[i]),
                    "risk_level": "High" if delay_probabilities[i] > 0.7 else "Medium" if delay_probabilities[i] > 0.4 else "Low"
                }
            )
            predictions.append(pred_response)

        response = BatchPredictionResponse(predictions=predictions)
        logger.info(f"Batch prediction made for {len(request.flights)} flights")
        return response

    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making batch prediction: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """
    Get information about the loaded models
    """
    if classification_model is None or regression_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")

    return {
        "classification_model_type": type(classification_model).__name__,
        "regression_model_type": type(regression_model).__name__,
        "feature_count": 15,  # Based on our feature engineering
        "timestamp": datetime.datetime.now().isoformat()
    }

if __name__ == "__main__":
    # For local development
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )