@echo off
echo Flight Delay Prediction Platform - Upload Script
echo.

REM This script will help you upload the project to GitHub
echo This script will NOT use your PAT directly for security reasons.
echo Instead, it guides you through the proper upload process.
echo.

REM Change to project directory
cd /d "%~dp0"

REM Initialize git if not already initialized
if not exist ".git" (
    echo Initializing Git repository...
    git init
)

REM Configure git (you can change these values as needed)
git config user.name "Your Name"
git config user.email "your.email@example.com"

REM Add all files to git
echo Adding all files to Git...
git add .

REM Create initial commit
git commit -m "Initial commit: Flight Delay Prediction Platform

Complete machine learning project with:
- FastAPI backend with CORS support
- Modern HTML/CSS/JavaScript frontend
- XGBoost models for flight delay prediction
- Docker configuration
- Training pipeline
- ETL data processing
"

REM Instructions for user
echo.
echo **********************************************
echo UPLOAD INSTRUCTIONS:
echo **********************************************
echo 1. Go to https://github.com/faiz0304/Flight-Delays-Predictor
echo 2. Copy the HTTPS clone URL: https://github.com/faiz0304/Flight-Delays-Predictor.git
echo 3. Run this command (replace with your actual URL):
echo    git remote add origin https://github.com/faiz0304/Flight-Delays-Predictor.git
echo 4. Then run:
echo    git branch -M main
echo    git push -u origin main
echo.
echo For security, please do NOT share your PAT token again.
echo If you already shared it, revoke it immediately in GitHub Settings.
echo **********************************************

REM Show status
git status

pause