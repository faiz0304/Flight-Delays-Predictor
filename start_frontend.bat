@echo off
REM Batch file to start the Flight Delay Prediction Streamlit frontend
echo Starting Flight Delay Prediction Frontend...
echo.

REM Change to the project directory
cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Start the Streamlit frontend
echo Starting Streamlit frontend on port 8501...
python -m streamlit run frontend/app.py --server.port 8501

pause