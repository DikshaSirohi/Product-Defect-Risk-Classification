# AI-Based Product Defect Risk Classification System

An end-to-end Machine Learning web application designed for demo purposes to predict and explain product defect risks (Low, Medium, High) in a 10-step manufacturing assembly line. Built with a **FastAPI backend**, a **scikit-learn HistGradientBoosting model** with **SHAP TreeExplainer local drivers**, and a custom **Streamlit interactive UI** incorporating dynamic process flow charts.

---

## 🚀 System Architecture

```text
defect_risk_app/
  backend/
    main.py                 # FastAPI core router and server routes
    config.py               # Paths, exclusions, and ML default constants
    schemas.py              # Pydantic payloads for predict/train
    train.py                # Command-line model retraining script
    services/
      data_service.py       # Data cleansing, duplicate dropping, imputation
      model_service.py      # ML Pipeline & aggregated SHAP local explainability
  frontend/
    app.py                  # High-aesthetic multi-tab Streamlit dashboard
  data/
    sample_defects.csv      # 5,000-row synthetic calibrated training set
  artifacts/
    defect_risk_model.joblib # Persisted joblib binary ML pipeline
  scripts/
    generate_sample_data.py # Data generation script implementing process rules
  tests/
    test_backend_smoke.py   # Automated REST endpoint smoke test suite
  requirements.txt          # Python package requirements
  docker-compose.yml        # Multi-container local orchestration
```

---

## 🛠️ Installation and Setup

### 1. Pre-requisites
Ensure Python 3.10+ is installed on your local machine.

### 2. Configure Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix/macOS:
source .venv/bin/activate

# Install required packages
python -m pip install -r requirements.txt
```

---

## ⚡ Running the Applications

### 1. Launch FastAPI Backend
The backend has an automated startup lifecycle. If no model is found in `artifacts/`, it automatically trains one from the default dataset at startup.
```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
*API documentation will be live at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)*

### 2. Launch Streamlit Frontend
Open a new terminal tab, activate the virtual environment, and run:
```bash
streamlit run frontend/app.py
```
*The Control Center dashboard will load automatically at: [http://localhost:8501](http://localhost:8501)*

---

## 📊 Retraining from CLI
To retrain the ML pipeline directly from the command line on any custom dataset:
```bash
python -m backend.train data/sample_defects.csv --target defect_risk
```

---

## 🧪 Automated Verification & Smoke Tests
Ensure the backend server is running on port 8000, then execute:
```bash
python tests/test_backend_smoke.py
```
*You should see all health, metadata, predictions, and SHAP tests output with standard checkmarks!*

---

## 🐳 Docker Deployment
To launch the entire backend and Streamlit frontend in a unified local container network:
```bash
docker-compose up --build
```
* Streamlit: `http://localhost:8501`
* FastAPI: `http://localhost:8000`
