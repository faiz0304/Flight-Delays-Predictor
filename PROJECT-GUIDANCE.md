# Flight Delay Prediction Platform - Developer Guidelines

## Coding Conventions

### Python Standards
- **Python version**: 3.10
- **Virtual environment**: Use `venv`
- **Naming conventions**: `snake_case` for variables/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- **Code formatting**: Follow PEP 8 standards

### API Response Format
```json
{
  "delay_probability": 0.78,
  "prediction": "Delayed"
}
```

### Error Handling
- Raise custom `HTTPException` in FastAPI with standard status codes (400, 404, 500)
- Use logging in JSON format with INFO level

### Environment Management
- Use `python-dotenv` to load `.env` file for environment variables
- Never commit `.env` file to git

### Git Workflow
- **Commit format**: Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`)
- **Branching strategy**: Trunk-based development (`main` branch protected, feature branches merged via PR)

### Code Organization
- Strict separation of concerns (ETL → Feature Store → ML Training → API)
- Use scikit-learn Pipelines to chain preprocessing and model execution to prevent data leakage

## Security Practices
- Never expose database credentials, MLflow artifacts bucket keys, or API keys in code
- Use HTTP Bearer headers for API validation
- Implement rate limiting with `slowapi` in FastAPI
- Use Pydantic models for strict input validation

## ML Pipeline Standards
- Target label `is_delayed` is strictly defined as an arrival delay > 15 minutes
- Use `is_delayed = 1 if arrival_delay > 15 else 0` as the target label
- Track all hyperparameters, metrics, and models in MLflow
- Maintain strict separation between training and inference code

## Performance Requirements
- Model ROC-AUC > 0.82 for classification tasks
- API inference latency < 100ms
- All dependencies in `requirements.txt` must be version-locked for deterministic builds