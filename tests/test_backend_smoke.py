import requests
import json
import sys
import time

def run_smoke_tests(base_url="http://127.0.0.1:8000"):
    print("=== Launching Backend Smoke Tests ===")
    
    # 1. Health check
    try:
        r = requests.get(f"{base_url}/health")
        assert r.status_code == 200, f"Health check failed with code {r.status_code}"
        health_data = r.json()
        assert health_data["status"] == "ok", "Health status is not 'ok'"
        print("[PASS] Health Endpoint")
    except Exception as e:
        print(f"[FAIL] Health Endpoint: ({e})")
        sys.exit(1)

    # 2. Metadata check
    try:
        r = requests.get(f"{base_url}/metadata")
        assert r.status_code == 200, f"Metadata failed with code {r.status_code}"
        metadata = r.json()
        assert "features" in metadata, "Missing features in metadata"
        assert "classes" in metadata, "Missing classes in metadata"
        print("[PASS] Metadata Endpoint")
    except Exception as e:
        print(f"[FAIL] Metadata Endpoint: ({e})")
        sys.exit(1)

    # 3. Model Card check
    try:
        r = requests.get(f"{base_url}/model-card")
        assert r.status_code == 200
        card = r.json()
        assert "intended_use" in card
        assert "metrics" in card
        print("[PASS] Model Card Endpoint")
    except Exception as e:
        print(f"[FAIL] Model Card Endpoint: ({e})")
        sys.exit(1)

    # 4. Predict endpoint check
    try:
        payload = {
            "record": {
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
        }
        r = requests.post(f"{base_url}/predict", json=payload)
        assert r.status_code == 200, f"Predict failed with code {r.status_code}: {r.text}"
        res = r.json()
        
        assert "prediction" in res, "Missing prediction class"
        assert res["prediction"] in ["Low", "Medium", "High"], f"Invalid prediction: {res['prediction']}"
        assert "confidence" in res, "Missing confidence value"
        assert "probabilities" in res, "Missing probability distributions"
        assert "explanation" in res, "Missing explanation metrics"
        
        # Test explanations format
        exp = res["explanation"]
        assert len(exp) > 0, "No explanations were calculated"
        assert "feature" in exp[0], "Explanation missing feature name"
        assert "contribution" in exp[0], "Explanation missing score"
        assert "effect" in exp[0], "Explanation missing process effect statement"
        
        print("[PASS] Single Prediction Endpoint")
        print(f"   -> Prediction: {res['prediction']} ({res['confidence']*100:.1f}% confidence)")
        print(f"   -> Key SHAP factor: '{exp[0]['feature']}' ({exp[0]['effect']})")
    except Exception as e:
        print(f"[FAIL] Single Prediction Endpoint: ({e})")
        sys.exit(1)

    # 5. Sample Data download check
    try:
        r = requests.get(f"{base_url}/sample-data.csv?limit=5")
        assert r.status_code == 200
        assert "text/csv" in r.headers.get("Content-Type", "")
        print("[PASS] Sample CSV Streaming Endpoint")
    except Exception as e:
        print(f"[FAIL] Sample CSV Streaming Endpoint: ({e})")
        sys.exit(1)

    print("\n* ALL ENDPOINT SMOKE TESTS PASSED *")

if __name__ == "__main__":
    url = "http://127.0.0.1:8000"
    if len(sys.argv) > 1:
        url = sys.argv[1]
    run_smoke_tests(url)
