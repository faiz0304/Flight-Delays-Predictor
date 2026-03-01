# Flight Delay Prediction Platform

## Project Overview

This project aims to build a production-oriented Flight Delay Prediction Platform to predict flight delays using historical aviation data from the U.S. Bureau of Transportation Statistics. The platform will provide a comprehensive solution for aviation delay prediction with both classification and regression capabilities.

## Business Value

Reducing reactive operational costs by enabling preemptive decision-making based on high-accuracy predictive modeling, achieving an expected ROC-AUC of ~0.82-0.86 using advanced tree-based models. The platform targets airline operations, travel platforms, and airport management to proactively manage scheduling, provide early delay warnings to passengers, and optimize gate allocations.

## Core Features

- Automated data ingestion from open-source aviation datasets
- Structured ETL pipeline for data cleaning and feature engineering
- End-to-end ML tracking and model registry
- High-performance, containerized inference API endpoint
- Continuous monitoring for data drift and operational health
- Interactive Streamlit frontend for user interaction

## Technical Architecture

- **Frontend**: Streamlit dashboard for interactive predictions
- **Backend**: FastAPI for high-throughput inference
- **ML Pipeline**: scikit-learn/XGBoost for modeling, MLflow for tracking
- **Database**: PostgreSQL for curated data and MLflow metadata
- **Deployment**: Dockerized application with container orchestration

## ML Objectives

- **Classification Task**: Predict if a flight will be delayed by more than 15 minutes (Binary: Yes/No)
- **Regression Task**: Predict the precise arrival delay in minutes

## Performance Goals

- API inference latency < 100ms
- Model ROC-AUC > 0.82
- Zero-downtime model updates