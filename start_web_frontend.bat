@echo off
REM Batch file to start the Flight Delay Prediction Web Frontend
echo Starting Flight Delay Prediction Web Frontend...
echo.

REM Change to the project directory
cd /d "%~dp0"

REM Kill any existing processes on port 8080
echo Stopping any existing processes on port 8080...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8080 ^| findstr LISTENING') do taskkill /f /pid %%a 2>nul

timeout /t 2 /nobreak >nul

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Start the web frontend server
echo Starting web server on port 8080...
echo Make sure the backend API is running at http://localhost:8000
echo.
python start_web_frontend.py

pause