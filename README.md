# AI-Based Product Defect Risk Classification System

An end-to-end Machine Learning web application designed for demo purposes to predict and explain product defect risks (Low, Medium, High) in a 10-step manufacturing assembly line. Built with a **scikit-learn HistGradientBoosting model** with **SHAP TreeExplainer local drivers**, and a custom **Streamlit interactive UI** incorporating dynamic process flow charts, fully self-contained as a single application.

---

## 🚀 System Architecture

```text
defect_risk_app/
  backend/
    config.py               # Paths, exclusions, and ML default constants
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
    test_pipeline.py        # Automated in-memory pipeline tests
  requirements.txt          # Python package requirements
  docker-compose.yml        # Streamlit container local orchestration
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

## ⚡ Running the Application

### Launch Streamlit Frontend
Activate the virtual environment, and run:
```bash
streamlit run frontend/app.py
```
*The Control Center dashboard will load automatically at: [http://localhost:8501](http://localhost:8501)*
On startup, it will automatically load the model (and retrain it if it doesn't find one).

---

## 📊 Retraining from CLI
To retrain the ML pipeline directly from the command line on any custom dataset:
```bash
python -m backend.train data/sample_defects.csv --target defect_risk
```

---

## 🧪 Automated Verification & Smoke Tests
To run automated in-memory pipeline tests:
```bash
python tests/test_pipeline.py
```

---

## 🐳 Docker Deployment
To launch the Streamlit container locally:
```bash
docker-compose up --build
```
* Streamlit: `http://localhost:8501`

