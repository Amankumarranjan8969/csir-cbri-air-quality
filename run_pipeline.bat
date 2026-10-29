@echo off
setlocal enabledelayedexpansion
title CSIR-CBRI Air Quality Forecasting System - One-Click Launcher

echo ===============================================================================
echo       CSIR-CBRI: MULTI-CITY AMBIENT AIR QUALITY FORECASTING SYSTEM
echo       Author: Aman Kumar Ranjan ^| Roll No: 23010301 ^| B.Tech CSE Data Science
echo       Partner: CSIR - Central Building Research Institute (CBRI), Roorkee
echo ===============================================================================
echo.

:: 1. Check Python Availability
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to your PATH.
    echo Please install Python 3.10+ from python.org and try again.
    pause
    exit /b 1
)

echo [Step 1/3] Python Environment Verified:
python --version
echo.

:: 2. Execute Model Training, Updating, and Verification Pipeline
echo [Step 2/3] Executing Model Training and Serialization Pipeline...
python run_pipeline.py
if %errorlevel% neq 0 (
    echo [WARNING] Pipeline run encountered an issue, but continuing to launch app...
)
echo.

:: 3. Launch Interactive Streamlit Web Application
echo [Step 3/3] Launching Interactive Web Dashboard (app.py)...
echo The application will automatically open in your default browser at http://localhost:8501
echo Press CTRL+C in this terminal window to stop the server when done.
echo.

streamlit run app.py --server.headless=false

pause
