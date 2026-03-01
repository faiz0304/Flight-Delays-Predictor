-- This is a placeholder for database initialization
-- In a real application, you would create tables here
CREATE TABLE IF NOT EXISTS flight_data (
    id SERIAL PRIMARY KEY,
    airline VARCHAR(10),
    origin VARCHAR(10),
    destination VARCHAR(10),
    distance INTEGER,
    scheduled_departure_hour INTEGER,
    arrival_delay FLOAT,
    is_delayed BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255),
    accuracy FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    roc_auc FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);