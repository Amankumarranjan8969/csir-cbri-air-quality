import os
import joblib
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="CSIR-CBRI Air Quality Forecasting API",
    version="2.0.0",
    description="Production REST API for validated PM2.5 forecasting, recursive multi-horizon predictions, and data validation."
)

ENGINE_PATH = "models/saved_models_v2/air_quality_inference_engine_v2.joblib"
engine = None

def get_engine():
    global engine
    if engine is None and os.path.exists(ENGINE_PATH):
        engine = joblib.load(ENGINE_PATH)
    return engine

class ObservationRecord(BaseModel):
    Date: str
    PM2_5: Optional[float] = None
    PM10: Optional[float] = None
    NO2: Optional[float] = None
    CO: Optional[float] = None
    Ozone: Optional[float] = None
    SO2: Optional[float] = None
    AT: Optional[float] = None
    RH: Optional[float] = None
    WS: Optional[float] = None
    BP: Optional[float] = None

class PredictionRequest(BaseModel):
    observations: List[ObservationRecord]
    model_choice: Optional[str] = "superblend"
    include_conformal_intervals: Optional[bool] = True

class ForecastRequest(BaseModel):
    horizon_days: Optional[int] = 7

@app.get("/health")
def health_check():
    eng = get_engine()
    return {
        "status": "healthy",
        "engine_loaded": eng is not None,
        "version": "2.0.0",
        "device": "NVIDIA GeForce RTX 4050 / CUDA"
    }

@app.get("/models")
def list_models():
    eng = get_engine()
    if eng is None:
        raise HTTPException(status_code=503, detail="Inference engine not loaded. Train models first.")
    return {
        "supported_models": list(eng.models_dict.keys()) + ["SuperBlend"],
        "default_model": "SuperBlend",
        "target": "PM2.5 (µg/m³)"
    }

@app.post("/predict")
def predict_pm25(req: PredictionRequest):
    eng = get_engine()
    if eng is None:
        raise HTTPException(status_code=503, detail="Inference engine not loaded. Train models first.")
    
    if not req.observations:
        raise HTTPException(status_code=400, detail="Observations list cannot be empty.")
        
    records = [r.dict() for r in req.observations]
    df_in = pd.DataFrame(records)
    
    try:
        results = eng.predict(df_in, model_choice=req.model_choice, return_intervals=req.include_conformal_intervals)
        return results.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/validate-data")
def validate_input_data(req: PredictionRequest):
    eng = get_engine()
    if eng is None:
        raise HTTPException(status_code=503, detail="Inference engine not loaded.")
    records = [r.dict() for r in req.observations]
    df_in = pd.DataFrame(records)
    try:
        val_df = eng.validate_input(df_in)
        return {
            "valid": True,
            "row_count": len(val_df),
            "date_range": [val_df['Date'].min().strftime('%Y-%m-%d'), val_df['Date'].max().strftime('%Y-%m-%d')],
            "columns_present": list(val_df.columns)
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}
