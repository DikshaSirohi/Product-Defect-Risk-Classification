import os

# Project Directories and Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_FILENAME = "defect_risk_model.joblib"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, MODEL_FILENAME)
SAMPLE_DATA_PATH = os.path.join(DATA_DIR, "sample_defects.csv")

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Default ML settings
DEFAULT_TARGET_COL = "defect_risk"
EXCLUDE_COLUMNS = [
    "product_id", 
    "timestamp", 
    "created_at", 
    "updated_at", 
    "id", 
    "Unnamed: 0",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
    "DefectStatus"
]
