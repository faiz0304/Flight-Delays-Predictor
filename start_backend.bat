@echo off
REM Batch file to start the Flight Delay Prediction API backend
echo Starting Flight Delay Prediction API Backend...
echo.

REM Change to the project directory
cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Kill any existing processes on port 8000
echo Stopping any existing processes on port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do taskkill /f /pid %%a 2>nul

timeout /t 2 /nobreak >nul

REM Start the FastAPI backend server with CORS support
echo Starting FastAPI server on port 8000 with CORS support...
uvicorn api.main:app --host 0.0.0.0 --port 8000

pause