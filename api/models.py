"""
Pydantic models for API input validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class FlightInput(BaseModel):
    """Input model for a single flight"""
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
    """Response model for prediction"""
    delay_probability: float = Field(..., example=0.78, description="Probability of delay > 15 minutes")
    prediction: str = Field(..., example="Delayed", description="Prediction result (Delayed/On Time)")
    delay_minutes: float = Field(..., example=25.5, description="Predicted delay in minutes")
    details: Dict[str, Any] = Field(..., description="Additional prediction details")

class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction"""
    flights: List[FlightInput]

class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction"""
    predictions: List[PredictionResponse]