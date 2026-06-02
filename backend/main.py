import os
import pandas as pd
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import SAMPLE_DATA_PATH, DEFAULT_TARGET_COL
from backend.schemas import (
    PredictionRequest, 
    PredictionResponse, 
    PredictionBatchRequest, 
    PredictionBatchResponse,
    ExplanationFactor
)
from backend.services.model_service import ModelService

app = FastAPI(
    title="AI-Based Product Defect Risk Classification API",
    description="Backend ML microservice for predicting and explaining manufacturing defect risks.",
    version="1.0.0"
)

# Enable CORS for local Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model service instance
model_service = ModelService()

@app.on_event("startup")
def startup_event():
    """
    On startup, load the pre-trained model artifact.
    If missing, automatically retrain using the default sample dataset.
    """
    print("=== Backend Startup Lifecycle ===")
    loaded = model_service.load_model()
    if not loaded:
        print("No pre-trained model artifact found. Initiating auto-training...")
        if os.path.exists(SAMPLE_DATA_PATH):
            try:
                df = pd.read_csv(SAMPLE_DATA_PATH)
                model_service.train_pipeline(df, DEFAULT_TARGET_COL)
                print("Auto-training completed successfully on startup!")
            except Exception as e:
                print(f"CRITICAL: Failed to auto-train model on startup: {e}")
        else:
            print(f"CRITICAL: Default training sample dataset not found at {SAMPLE_DATA_PATH}")

@app.get("/health")
def get_health():
    """
    Returns server health and model load status.
    """
    return {
        "status": "ok",
        "model_loaded": model_service.is_model_loaded(),
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/metadata")
def get_metadata():
    """
    Returns preprocessor structure, features, categorical categories, and default means/modes.
    """
    if not model_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model is not available. Retrain the model.")
    return model_service.get_metadata()

@app.get("/metrics")
def get_metrics():
    """
    Returns the accuracy, precision, recall, F1, and confusion matrix of the trained model.
    """
    if not model_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model is not available.")
    return model_service.get_metadata().get("metrics", {})

@app.get("/sample-data")
def get_sample_data(limit: int = 10):
    """
    Returns sample dataset rows as a JSON list.
    """
    if not os.path.exists(SAMPLE_DATA_PATH):
        raise HTTPException(status_code=404, detail="Sample dataset not found.")
    try:
        df = pd.read_csv(SAMPLE_DATA_PATH).head(limit)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read sample data: {e}")

@app.get("/sample-data.csv")
def get_sample_data_csv(limit: int = 250):
    """
    Returns sample dataset rows as a downloadable CSV attachment.
    """
    if not os.path.exists(SAMPLE_DATA_PATH):
        raise HTTPException(status_code=404, detail="Sample dataset not found.")
    try:
        df = pd.read_csv(SAMPLE_DATA_PATH).head(limit)
        csv_content = df.to_csv(index=False)
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=sample_defects_{limit}.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compile sample CSV: {e}")

@app.post("/train")
async def train_model(file: Optional[UploadFile] = File(None), target_col: str = Form(DEFAULT_TARGET_COL)):
    """
    Triggers model retraining.
    If a custom CSV is uploaded, it trains on it. Otherwise, it uses the default sample dataset.
    """
    if file:
        try:
            df = pd.read_csv(file.file)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse uploaded CSV: {e}")
    else:
        if not os.path.exists(SAMPLE_DATA_PATH):
            raise HTTPException(status_code=404, detail="No file uploaded and default sample data missing.")
        df = pd.read_csv(SAMPLE_DATA_PATH)

    try:
        metadata = model_service.train_pipeline(df, target_col)
        return {
            "status": "success",
            "message": "Model trained and serialized successfully.",
            "metrics": metadata.get("metrics"),
            "trained_at": metadata.get("trained_at")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {e}")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Performs real-time risk classification for a single lot or assembly row.
    """
    if not model_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    try:
        result = model_service.predict_record(request.record)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-json-batch", response_model=PredictionBatchResponse)
def predict_json_batch(request: PredictionBatchRequest):
    """
    Scores multiple JSON records in a single payload.
    """
    if not model_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    try:
        results = []
        for rec in request.records:
            pred_res = model_service.predict_record(rec)
            results.append(pred_res)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch-predict")
def batch_predict(file: UploadFile = File(...)):
    """
    Scores an uploaded CSV batch. Appends predictions and confidences,
    preserving all original extra columns, and returns a scored CSV attachment.
    """
    if not model_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded CSV: {e}")

    predictions = []
    confidences = []
    
    # Process row by row
    for _, row in df.iterrows():
        record = row.to_dict()
        try:
            pred_res = model_service.predict_record(record)
            predictions.append(pred_res["prediction"])
            confidences.append(pred_res["confidence"])
        except Exception:
            # Fallback if prediction fails on a single noisy row
            predictions.append("Error")
            confidences.append(0.0)

    # Append scores
    df["predicted_defect_risk"] = predictions
    df["prediction_confidence"] = confidences

    # Convert to CSV response
    csv_content = df.to_csv(index=False)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=scored_manufacturing_defects.csv"}
    )

@app.get("/model-card")
def get_model_card():
    """
    Returns standard model card information: goals, target variables, limitations, and explainability statements.
    """
    metrics = {}
    trained_at = "Not trained"
    if model_service.is_model_loaded():
        metadata = model_service.get_metadata()
        metrics = metadata.get("metrics", {})
        trained_at = metadata.get("trained_at", "N/A")

    return {
        "model_name": "AI-Based Product Defect Risk Classifier",
        "intended_use": (
            "This model provides decision support for quality engineers and supervisors. "
            "It is designed to screen production settings and inspection scores for elevated defect risks."
        ),
        "limitations": (
            "Not certified for direct automated safety-critical actions. "
            "Model operates on tabular parameters and inspection data; it does not replace downstream manual final testing."
        ),
        "target_variable": DEFAULT_TARGET_COL,
        "classes": ["Low", "Medium", "High"],
        "primary_algorithm": "RandomForestClassifier",
        "class_weighting": "balanced",
        "trained_date": trained_at,
        "metrics": metrics,
        "explainability_statement": (
            "Local predictions are evaluated using SHAP TreeExplainer to aggregate one-hot categorical "
            "contributions back to original raw features. In mathematical edge cases, standard model "
            "importances act as global fallback coefficients."
        )
    }
