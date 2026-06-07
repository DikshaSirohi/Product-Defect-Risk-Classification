import os
import sys
import pandas as pd
import numpy as np

# Ensure workspace root is in path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from backend.config import SAMPLE_DATA_PATH, DEFAULT_TARGET_COL
from backend.services.model_service import ModelService

def run_pipeline_tests():
    print("=== Launching ML Pipeline In-Memory Tests ===")
    
    # 1. Initialize and load service
    try:
        service = ModelService()
        # Verify the model can load, or if missing we will train it
        loaded = service.load_model()
        print(f"[PASS] Model Service Initialization (Loaded: {loaded})")
    except Exception as e:
        print(f"[FAIL] Model Service Initialization: ({e})")
        sys.exit(1)

    # 2. Retraining check
    try:
        print("Testing pipeline retraining...")
        assert os.path.exists(SAMPLE_DATA_PATH), f"Sample data missing at {SAMPLE_DATA_PATH}"
        df = pd.read_csv(SAMPLE_DATA_PATH)
        metadata = service.train_pipeline(df, DEFAULT_TARGET_COL)
        
        assert service.is_model_loaded(), "Model is not loaded after training"
        assert "metrics" in metadata, "Missing metrics in trained metadata"
        assert "features" in metadata, "Missing features in trained metadata"
        assert "classes" in metadata, "Missing classes in trained metadata"
        print(f"[PASS] Pipeline Retraining (Accuracy: {metadata['metrics']['accuracy']:.4f})")
    except Exception as e:
        print(f"[FAIL] Pipeline Retraining: ({e})")
        sys.exit(1)

    # 3. Single Prediction & Explanation check
    try:
        record = {
            "product_type": "motor",
            "line_id": "line_3",
            "shift": "night",
            "supplier_grade": "C",
            "inspection_method": "camera",
            "machine_temperature_c": 98.5,
            "vibration_mm_s": 5.4,
            "humidity_pct": 62.0,
            "pressure_bar": 6.1,
            "conveyor_speed_m_min": 31.5,
            "material_batch_age_days": 45,
            "operator_experience_months": 18,
            "maintenance_days_since": 72,
            "inspection_score": 52.5,
            "surface_roughness_um": 4.8
        }
        res = service.predict_record(record)
        
        assert "prediction" in res, "Missing prediction class"
        assert res["prediction"] in ["Low", "Medium", "High"], f"Invalid prediction: {res['prediction']}"
        assert "confidence" in res, "Missing confidence value"
        assert "probabilities" in res, "Missing probability distributions"
        assert "explanation" in res, "Missing explanation metrics"
        
        exp = res["explanation"]
        assert len(exp) > 0, "No explanations were calculated"
        assert "feature" in exp[0], "Explanation missing feature name"
        assert "contribution" in exp[0], "Explanation missing score"
        assert "effect" in exp[0], "Explanation missing process effect statement"
        
        print("[PASS] Single Prediction and SHAP explanation in-memory")
        print(f"   -> Prediction: {res['prediction']} ({res['confidence']*100:.1f}% confidence)")
        print(f"   -> Key SHAP factor: '{exp[0]['feature']}' ({exp[0]['effect']})")
    except Exception as e:
        print(f"[FAIL] Single Prediction and Explanation: ({e})")
        sys.exit(1)

    # 4. Batch prediction simulations
    try:
        df_batch = df.head(10).copy()
        
        predictions = []
        confidences = []
        for _, row in df_batch.iterrows():
            rec = row.to_dict()
            pred_res = service.predict_record(rec)
            predictions.append(pred_res["prediction"])
            confidences.append(pred_res["confidence"])
            
        df_batch["predicted_defect_risk"] = predictions
        df_batch["prediction_confidence"] = confidences
        
        assert "predicted_defect_risk" in df_batch.columns, "Predictions column missing"
        assert "prediction_confidence" in df_batch.columns, "Confidences column missing"
        assert len(df_batch) == 10, "Row count changed during scoring"
        print("[PASS] Batch scoring in-memory simulation")
    except Exception as e:
        print(f"[FAIL] Batch scoring simulation: ({e})")
        sys.exit(1)

    print("\n* ALL ML PIPELINE TESTS PASSED *")

if __name__ == "__main__":
    run_pipeline_tests()
