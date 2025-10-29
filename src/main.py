# src/main.py
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# Relative path to the saved model file
MODEL_PATH = "sarima_model.joblib"

# --- API Schemas ---
class ForecastQuery(BaseModel):
    """Input schema for the prediction endpoint."""
    steps: int = 12  # Number of months to forecast
    
class ForecastResult(BaseModel):
    """Output schema for the prediction endpoint."""
    forecast_start_date: str
    forecast_end_date: str
    predictions: Dict[str, float]

# --- FastAPI App Initialization ---
app = FastAPI(
    title="CO2 Time Series Forecaster",
    description="SARIMA model serving monthly CO2 predictions.",
    version="1.0.0",
)

# Global model variable to hold the loaded model
model_fit = None

# --- Startup Event: Load Model ---
@app.on_event("startup")
async def load_model():
    """Loads the trained SARIMA model when the server starts."""
    global model_fit
    try:
        model_fit = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # In a real app, this should be a critical failure
        raise RuntimeError("Failed to load model on startup.")

# --- API Endpoints ---

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"status": "Model service running", "model_loaded": model_fit is not None}

@app.post("/forecast", response_model=ForecastResult)
async def get_forecast(query: ForecastQuery):
    """Generates a time series forecast for the next 'steps' months."""
    if model_fit is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or ready.")

    try:
        # Generate the forecast
        forecast = model_fit.get_forecast(steps=query.steps)
        
        # Extract predictions and index dates
        predictions_series = forecast.predicted_mean
        
        # Format output
        predictions_dict = {
            date.strftime('%Y-%m-%d'): round(value, 2)
            for date, value in predictions_series.items()
        }
        
        return ForecastResult(
            forecast_start_date=predictions_series.index.min().strftime('%Y-%m-%d'),
            forecast_end_date=predictions_series.index.max().strftime('%Y-%m-%d'),
            predictions=predictions_dict
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# To test locally: 
# uvicorn src.main:app --reload