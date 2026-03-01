# PROJECT-CONTEXT.md

## 1. Project Overview

### Problem Statement

Aviation delays cause significant logistical and financial disruptions. This project aims to build a production-oriented Flight Delay Prediction Platform to predict flight delays using historical aviation data from the U.S. Bureau of Transportation Statistics.

### Target Users

* **Airline Operations:** To proactively manage fleet scheduling.
* **Travel Platforms:** To provide early delay warnings to passengers.
* **Airport Management:** To optimize gate allocations and ground crew deployments.

### Core Features

* Automated data ingestion from open-source aviation datasets.


* Structured ETL pipeline for data cleaning and feature engineering.


* End-to-end ML tracking and model registry.


* High-performance, containerized inference API endpoint.


* Continuous monitoring for data drift and operational health.



### ML Objective

* 
**Classification Task:** Predict if a flight will be delayed by more than 15 minutes (Binary: Yes/No).


* 
**Regression Task:** Predict the precise arrival delay in minutes.



### Business Value

Reducing reactive operational costs by enabling preemptive decision-making based on high-accuracy predictive modeling, achieving an expected ROC-AUC of ~0.82-0.86 using advanced tree-based models.

### High-Level Architecture

1. **Frontend:** Streamlit dashboard for interactive predictions.
2. 
**Backend:** FastAPI for high-throughput inference.


3. 
**ML Pipeline:** scikit-learn/XGBoost for modeling , MLflow for tracking.


4. 
**Database:** PostgreSQL for curated data and MLflow metadata.


5. 
**Deployment:** Dockerized application deployed via GitHub Actions to GCP Cloud Run.



### Scalability Vision

The system architecture transitions from local pandas operations to stateless containerized microservices. The inference API will horizontally scale using GCP Cloud Run to handle highly concurrent request volumes.

### Security Vision

Environment variables for secrets, strict IAM roles for cloud services, and validated input via Pydantic to prevent injection attacks.

### Performance Goals

* API inference latency `< 100ms`.
* Model ROC-AUC `> 0.82`.


* Zero-downtime model updates.



### Constraints and Assumptions

* Relies on the public U.S. Bureau of Transportation Statistics dataset.


* Target label `is_delayed` is strictly defined as an arrival delay $> 15$ minutes.



---

## 2. Technology Stack

### Frontend

* **Framework:** Streamlit. (Optimized for rapid ML application development without managing separate React/Vue assets).
* **State management:** Streamlit Session State.
* **Styling system:** Native Streamlit components.
* **HTTP client:** `requests` library.

### Backend

* 
**Framework:** FastAPI. (Provides native asynchronous support, automatic OpenAPI documentation, and strict data validation).


* **API design style:** REST.
* **Async or Sync:** Async.
* **Dependency management:** `pip` with `requirements.txt`.

### ML Layer

* 
**Model type:** XGBoost. (Yields highest baseline ROC-AUC for tabular flight data ).


* 
**Training framework:** scikit-learn.


* 
**Model versioning:** MLflow.


* 
**Serialization format:** Joblib.


* 
**Experiment tracking:** MLflow (Parameters, Metrics, Artifacts).



### Database

* 
**Type:** PostgreSQL. (Robust ACID compliance for both structured analytical data and MLflow backend).


* 
**ORM:** SQLAlchemy.


* **Migration tool:** Alembic.

### Caching

* **Tool:** Redis (Optional, for caching frequent identical predictions).

### Authentication

* **Tool:** API Keys via FastAPI `Security` headers (Simple, stateless authorization for microservices).

### Background Jobs

* 
**Tool:** Apache Airflow. (For scheduling monthly data pulls and automated retraining workflows ).



### DevOps

* 
**Containerization:** Docker.


* 
**CI/CD:** GitHub Actions.


* 
**Hosting:** GCP Cloud Run. (Serverless, scales to zero, perfect for stateless API endpoints).


* 
**Model hosting:** Packaged within Docker container via FastAPI.


* 
**Monitoring:** Evidently AI for data drift.


* **Logging:** Python `logging` module natively integrated with GCP Cloud Logging.

### Why this stack is optimized for:

* **Hackathon speed:** FastAPI and Streamlit allow full-stack ML deployment in hours.
* **Production scalability:** Cloud Run and PostgreSQL handle massive traffic automatically.
* **ML integration:** MLflow seamlessly registers XGBoost models into the FastAPI service.
* **Low operational complexity:** Serverless deployment removes infrastructure management overhead.

---

## 3. Directory Structure

```text
flight_delay_project/
[cite_start]├── data/                  # Local storage for raw/processed CSVs [cite: 117]
[cite_start]├── etl/                   # Extraction and transformation scripts [cite: 118]
[cite_start]├── features/              # Feature engineering logic (distance, encoding) [cite: 119]
[cite_start]├── models/                # Training pipelines and hyperparameter tuning [cite: 120]
[cite_start]├── evaluation/            # Validation scripts, metrics calculation [cite: 121]
[cite_start]├── api/                   # FastAPI backend implementation [cite: 122]
[cite_start]├── mlflow/                # MLflow tracking backend configurations [cite: 123]
├── frontend/              # Streamlit application UI
├── tests/                 # Unit and integration tests (Pytest)
├── scripts/               # Bash utilities for setup and DB migrations
├── infra/                 # Terraform/Cloud deployment configurations
├── docker/                # Dockerfiles and docker-compose.yml
[cite_start]├── .github/workflows/     # CI/CD pipelines (GitHub Actions) [cite: 158]
[cite_start]├── requirements.txt       # Python dependencies [cite: 152]
└── .env                   # Environment variables template

```

**Directory Purposes:**

* 
`data/`, `etl/`, `features/`, `models/`, `evaluation/`, `api/`, `mlflow/`: Standard modular ML pipeline structure.


* `frontend/`: Separates user interface logic.
* 
`docker/`: Centralizes container build context.


* 
`.github/workflows/`: Automates code linting, testing, and deployment.



---

## 4. Coding Conventions

* 
**Python version:** `3.10`.


* **Virtual environment setup:** `venv`.
* **Naming conventions:** `snake_case` for variables/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
* **API response format standard:**
```json
{
  "delay_probability": 0.78,
  "prediction": "Delayed"
}

```





* **Error handling standard:** Raise custom `HTTPException` in FastAPI with standard status codes (400, 404, 500).
* **Logging standard:** Standardized JSON format using Python's `logging`, set to `INFO` level.
* **Environment variable management:** `python-dotenv` loading `.env` file (never committed to git).
* **Git commit format:** Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`).
* **Branching strategy:** Trunk-based development (`main` branch protected, feature branches merged via PR).
* 
**Code modularization rules:** Strict separation of concerns (ETL $\rightarrow$ Feature Store $\rightarrow$ ML Training $\rightarrow$ API).


* **ML pipeline structure standard:** Use `scikit-learn` Pipelines to chain preprocessing and model execution to prevent data leakage.

---

## 5. Key Commands

**Environment Setup:**

```bash
python3.10 -m venv venv
source venv/bin/activate

```

**Installing Dependencies:**

```bash
pip install -r requirements.txt

```

**Running MLflow Server Locally:**

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow/artifacts

```

**Training Model:**

```bash
python -m models.train_xgboost

```

**Running Backend:**

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

```

**Running Frontend:**

```bash
streamlit run frontend/app.py

```

**Running Full Stack Locally (Docker):**

```bash
docker-compose up --build

```

**Running Migrations:**

```bash
alembic upgrade head

```

**Linting and Formatting:**

```bash
flake8 .
black .

```

**Running Tests:**

```bash
pytest tests/

```

---

## 6. Important Notes

* **Security warnings:** Never expose database credentials, MLflow artifacts bucket keys, or API keys in code. Use GCP Secret Manager in production.
* **API key handling:** Use HTTP Bearer headers for API validation.
* 
**Model performance considerations:** XGBoost performs best (~0.82-0.86 ROC-AUC) but requires careful monitoring of feature engineering depth to prevent overfitting.


* **Data privacy notes:** Flight data from BTS is public domain, but ensure any joined user-identifiable data (e.g., passenger queries) is anonymized.
* **Rate limiting:** Implement `slowapi` in FastAPI to prevent endpoint abuse.
* 
**Validation standards:** Use Pydantic models for strict input validation (e.g., airline strings, distance bounds).


* 
**Common failure points:** Data schema changes from BTS open-source files breaking the ETL pipeline.


* 
**Debugging approach:** Cross-reference inference logs with MLflow experiment runs  to diagnose model degradation.



---

## 7. End-to-End Workflow

1. 
**Problem definition:** Predict if arrival delay $> 15$ mins.


2. 
**Data collection:** Download raw CSV from BTS.


3. 
**Data cleaning:** Remove null delays, convert categorical columns.


4. 
**Feature engineering:** Engineer day of week, month, route distance buckets, airline encoding.


5. 
**Model training:** Train Logistic Regression, Random Forest, and XGBoost models.


6. 
**Evaluation:** Calculate Accuracy, Precision, Recall, F1-score, and ROC-AUC.


7. 
**Serialization:** Save the best model (based on ROC-AUC) using Joblib or Pickle.


8. **Backend integration:** Load the serialized model into the FastAPI application state.
9. 
**API exposure:** Expose a `POST /predict` endpoint accepting JSON input.


10. **Frontend integration:** Connect the Streamlit UI to the FastAPI endpoint.
11. 
**Testing:** Run Pytest on ETL scripts, API endpoints, and model inference schemas.


12. 
**Containerization:** Write a Dockerfile extending `python:3.10`, copy requirements, and configure `uvicorn`.


13. 
**CI/CD:** Trigger GitHub Actions on push to lint, test, build, and push Docker image to cloud registry.


14. 
**Production deployment:** Deploy container to GCP Cloud Run.


15. 
**Monitoring:** Log inference requests and track prediction distributions using Evidently AI.


16. 
**Iteration loop:** Schedule Airflow DAG to pull new monthly data, retrain the model, register the new version, and auto-update the API.



---

## 8. Verification and Testing

* **Unit testing:** `pytest` for ETL functions (data cleaning correctness, feature generation logic).
* **Integration testing:** `pytest-asyncio` and `httpx` to verify FastAPI request/response flow.
* 
**ML validation:** scikit-learn cross-validation  ensuring validation ROC-AUC matches test ROC-AUC.


* **API contract testing:** OpenAPI schemas validated automatically via FastAPI.
* **Load testing:** `locust` to simulate high concurrent user queries on the prediction endpoint.
* **Security testing:** `bandit` for static Python security scanning.
* 
**Model drift monitoring:** Evidently AI dashboard to detect feature distribution shifts over time.


* 
**Acceptance criteria:** Model ROC-AUC $> 0.82$, API latency $< 100$ms, complete CI/CD pipeline deployment to cloud.



---

## 9. Deployment

* 
**Hosting provider:** Google Cloud Platform (GCP).


* **Frontend deployment:** GCP Cloud Run (Serverless UI).
* 
**Backend deployment:** GCP Cloud Run (Serverless API).


* 
**Database deployment:** Cloud SQL for PostgreSQL.


* 
**Model deployment:** Embedded within the Backend Docker container.


* **Environment configuration:** GCP Secret Manager injected as environment variables at runtime.
* **Scaling strategy:** Cloud Run horizontal autoscaling (scales out based on concurrent HTTP requests).
* 
**Zero-downtime strategy:** Cloud Run traffic splitting (Canary deployment), routing 100% of traffic to the new revision only once health checks pass.


* **Backup strategy:** Automated daily Cloud SQL backups and model artifact versioning in GCP Cloud Storage.
* **Rollback plan:** Single-click traffic reroute to previous Cloud Run revision via GitHub Actions workflow or GCP Console.

---

## 10. Dos and Don'ts

### DO

* Encapsulate all feature transformations inside a robust scikit-learn Pipeline to guarantee identical processing during training and inference.
* Lock all dependency versions in `requirements.txt` to ensure deterministic Docker builds.


* Use `is_delayed = 1 if arrival_delay > 15 else 0` strictly as the target label.


* Track all hyperparameters, metrics, and models in MLflow.



### DON'T

* Do not train the model directly in the API script; maintain strict separation of concerns.


* Do not leak the target variable (`arrival_delay`) into the features used for prediction.
* Do not deploy Docker containers as the `root` user; always create a non-root user in the Dockerfile for security.
* Do not hardcode cloud credentials in GitHub Actions; use OIDC or encrypted GitHub Secrets.