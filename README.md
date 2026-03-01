# Flight Delay Prediction Platform

## Project Overview

This project implements a production-oriented Flight Delay Prediction Platform to predict flight delays using historical aviation data. The platform provides both classification (delayed or not) and regression (exact delay minutes) capabilities.

## Architecture

The system follows a microservices architecture with:

- **Frontend**: Streamlit dashboard for interactive predictions
- **Backend**: FastAPI for high-throughput inference
- **ML Pipeline**: scikit-learn/XGBoost for modeling, MLflow for tracking
- **Database**: PostgreSQL for curated data and MLflow metadata
- **Deployment**: Dockerized application with container orchestration

## Features

- Automated data ingestion from open-source aviation datasets
- Structured ETL pipeline for data cleaning and feature engineering
- End-to-end ML tracking and model registry
- High-performance, containerized inference API endpoint
- Continuous monitoring for data drift and operational health
- Interactive Streamlit frontend for user interaction

## Models

- **Classification Model**: Predicts if a flight will be delayed by more than 15 minutes (Binary: Yes/No)
- **Regression Model**: Predicts the precise arrival delay in minutes

## Performance Goals

- API inference latency < 100ms
- Model ROC-AUC > 0.82
- Zero-downtime model updates

## Project Status

**PHASE 1: Environment & Scaffolding** ✅ COMPLETE
- Directory structure created
- Dependencies installed
- Virtual environment set up

**PHASE 2: Data & ML Pipeline MVP** ✅ COMPLETE
- Synthetic data generator created
- ETL preprocessing pipeline built
- XGBoost models trained for both classification and regression
- MLflow integration for experiment tracking

**PHASE 3: Backend API** ✅ COMPLETE
- FastAPI application built
- Pydantic input validation implemented
- POST /predict endpoint created
- API tested and verified

**PHASE 4: Frontend UI** ✅ COMPLETE
- Streamlit application built
- Prediction form created
- Connected to API backend
- UI functionality tested

**PHASE 5: Containerization** ✅ COMPLETE
- Multi-stage Dockerfile for backend
- Dockerfile for Streamlit frontend
- Docker-compose.yml for full stack deployment

## Directory Structure

```
flight_delay_project/
├── data/                  # Raw and processed datasets
├── etl/                   # ETL scripts and preprocessing logic
├── features/              # Feature engineering logic
├── models/                # Training pipelines and model artifacts
├── evaluation/            # Validation and metrics calculation
├── api/                   # FastAPI backend implementation
├── mlflow/                # MLflow tracking configurations
├── frontend/              # Streamlit application
├── docker/                # Dockerfiles and docker-compose.yml
├── scripts/               # Utility scripts
├── tests/                 # Unit and integration tests
└── requirements.txt       # Python dependencies
```

## Running the Application

### Option 1: Web Frontend (Recommended) - Easiest Method

1. **Start both services** (recommended):
   ```bash
   start_both.bat
   ```

2. **Access the application**:
   - Backend API: `http://localhost:8000`
   - Web Frontend: `http://localhost:8080`

### Option 2: Manual Start

1. Start the backend API with CORS support:
   ```bash
   start_backend.bat
   ```

2. In a new terminal, start the web frontend:
   ```bash
   start_web_frontend.bat
   ```

3. Open your browser and go to `http://localhost:8080`

### Option 3: Streamlit Frontend (Legacy)

1. Start the backend API:
   ```bash
   start_backend.bat
   ```

2. Start the Streamlit frontend in a new terminal:
   ```bash
   start_frontend.bat
   ```

3. Access the frontend at `http://localhost:8501`

### Option 4: Docker Compose

To run the full application stack with Docker:

```bash
docker-compose -f docker/docker-compose.yml up --build
```

### Web Frontend Features

The new web frontend provides:

- **Modern UI**: Clean, responsive design with gradient backgrounds
- **Real-time Predictions**: Instant flight delay predictions via API
- **Risk Visualization**: Interactive charts showing risk factors
- **Flight Information**: Detailed display of flight parameters
- **Risk Assessment**: Color-coded risk levels (Low/Medium/High)
- **Mobile Responsive**: Works on all device sizes
- **CORS Support**: Seamless integration with backend API

Access the web frontend at `http://localhost:8080` after starting the backend API.

### Troubleshooting

- If you get port binding errors, use the batch files which automatically handle port conflicts
- Make sure to run both backend and frontend services for full functionality
- The batch files will automatically terminate existing processes on ports 8000 and 8080

### Docker Compose

To run the full application stack with Docker:

```bash
docker-compose -f docker/docker-compose.yml up --build
```

## API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /predict` - Single flight delay prediction
- `POST /predict/batch` - Batch flight delay prediction
- `GET /model-info` - Model information

## API Example

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
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
}'
```

## Model Performance

Current model performance metrics:
- Classification ROC-AUC: ~0.68 (target: >0.82)
- Regression RMSE: ~21 min (room for improvement)

## Future Improvements

- Hyperparameter tuning to improve model performance
- Feature engineering enhancements
- Advanced ensemble methods
- Real-time data integration
- Model monitoring and drift detection
- Automated retraining pipeline

## Security Considerations

- Environment variables for secrets management
- Input validation through Pydantic models
- Non-root containers for security
- API rate limiting (can be added with slowapi)

## Technologies Used

- Python 3.10
- FastAPI
- Streamlit
- XGBoost
- scikit-learn
- MLflow
- PostgreSQL
- Docker
- Docker Compose