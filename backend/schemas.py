from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class PredictionRequest(BaseModel):
    record: Dict[str, Any] = Field(
        ..., 
        example={
            "product_type": "motor",
            "line_id": "line_3",
            "shift": "night",
            "supplier_grade": "B",
            "inspection_method": "camera",
            "machine_temperature_c": 84.2,
            "vibration_mm_s": 4.8,
            "humidity_pct": 62.0,
            "pressure_bar": 6.1,
            "conveyor_speed_m_min": 31.5,
            "material_batch_age_days": 45,
            "operator_experience_months": 18,
            "maintenance_days_since": 24,
            "inspection_score": 76.5,
            "surface_roughness_um": 3.2
        }
    )

class ExplanationFactor(BaseModel):
    feature: str
    method: str
    contribution: float
    effect: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    explanation: List[ExplanationFactor]

class PredictionBatchRequest(BaseModel):
    records: List[Dict[str, Any]]

class PredictionBatchResponse(BaseModel):
    predictions: List[PredictionResponse]

class TrainRequest(BaseModel):
    target_col: str = "defect_risk"
