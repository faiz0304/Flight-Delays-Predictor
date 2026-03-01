@echo off
REM Batch file to start both backend and web frontend services
echo Starting Flight Delay Prediction Platform (Both Services)...
echo.

REM Change to the project directory
cd /d "%~dp0"

REM Kill any existing processes on ports 8000 and 8080
echo Stopping any existing processes on ports 8000 and 8080...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do taskkill /f /pid %%a 2>nul
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8080 ^| findstr LISTENING') do taskkill /f /pid %%a 2>nul

timeout /t 3 /nobreak >nul

REM Start backend in a new window
echo Starting backend API server with CORS support...
start "Flight Delay API" cmd /c "venv\Scripts\activate.bat && uvicorn api.main:app --host 0.0.0.0 --port 8000"

REM Wait a moment for backend to start
timeout /t 5 /nobreak >nul

REM Start web frontend in a new window
echo Starting web frontend...
start "Flight Delay Web Frontend" cmd /c "venv\Scripts\activate.bat && python start_web_frontend.py"

echo.
echo Both services are starting...
echo - Backend API: http://localhost:8000
echo - Web Frontend: http://localhost:8080
echo.
echo Closing this window will not stop the services.
pause