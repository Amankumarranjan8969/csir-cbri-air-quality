# ===============================================================================
# CSIR-CBRI: MULTI-CITY AMBIENT AIR QUALITY FORECASTING SYSTEM
# Author: Aman Kumar Ranjan | Roll No: 23010301 | B.Tech CSE Data Science
# Partner: CSIR - Central Building Research Institute (CBRI), Roorkee
# ===============================================================================

Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "      CSIR-CBRI: MULTI-CITY AMBIENT AIR QUALITY FORECASTING SYSTEM" -ForegroundColor Yellow
Write-Host "      Author: Aman Kumar Ranjan | Roll No: 23010301 | B.Tech CSE Data Science" -ForegroundColor White
Write-Host "      Partner: CSIR - Central Building Research Institute (CBRI), Roorkee" -ForegroundColor Green
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python
try {
    $pyVer = python --version 2>&1
    Write-Host "[Step 1/3] Python Detected: $pyVer" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH." -ForegroundColor Red
    pause
    exit 1
}

# 2. Run Pipeline
Write-Host "`n[Step 2/3] Training, updating and serializing models across all cities..." -ForegroundColor Cyan
python run_pipeline.py

# 3. Launch Streamlit
Write-Host "`n[Step 3/3] Launching Streamlit Web Application..." -ForegroundColor Cyan
Write-Host "Opening in browser at http://localhost:8501 (Press CTRL+C to stop)..." -ForegroundColor Yellow
streamlit run app.py --server.headless=false
