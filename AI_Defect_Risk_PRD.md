# Product Requirements Document: AI-Based Product Defect Risk Classification

## 1. Product Summary

### Product Name
AI-Based Product Defect Risk Classification

### Purpose
Build an end-to-end machine learning application that classifies product defect risk in a manufacturing process using production parameters, inspection readings, material context, and operational metadata. The product helps quality assurance teams prioritize inspections, reduce scrap, detect high-risk process conditions earlier, and create explainable quality-improvement recommendations.

### Primary Deliverables
1. Python backend API for model training, prediction, batch scoring, metrics, and explainability.
2. Streamlit frontend for quality engineers and manufacturing stakeholders.
3. Reproducible sample dataset and training workflow.
4. End-to-end documentation covering product scope, feature design, implementation, deployment, and acceptance criteria.

### Business Outcome
Reduce avoidable waste and rework by surfacing defect risk before or during inspection, allowing operators and quality engineers to act earlier.

---

## 2. Problem Statement

Manufacturing quality teams often rely on manual inspection, rule-based thresholds, or delayed defect reporting. These methods can miss complex interactions between process settings, material batches, shifts, machine behavior, and inspection scores. The result is late detection, unnecessary scrap, slow root-cause analysis, and inconsistent quality decisions.

This product predicts whether a product, lot, or production window has **Low**, **Medium**, or **High** defect risk. It also explains the most influential factors behind each prediction so quality engineers can take operational action instead of treating the model as a black box.

---

## 3. Goals and Non-Goals

### Goals
- Train a defect-risk classification model from labeled manufacturing data.
- Support single-record prediction for real-time operator use.
- Support CSV batch prediction for quality review and offline analysis.
- Display model performance metrics, confusion matrix, and feature importance.
- Provide explainability using SHAP when available, with a fallback to global model importance.
- Provide a simple Streamlit UI that can be used by non-technical users.
- Make the project runnable locally with minimal setup.
- Make the code modular enough for future production deployment.

### Non-Goals for Version 1
- Real-time machine integration through PLC, SCADA, MES, or OPC-UA.
- Automated line stoppage or automated rejection decisions.
- Regulatory-grade validation for safety-critical manufacturing.
- Full MLOps platform with automated drift detection, feature store, and CI/CD model registry.
- Deep learning image defect detection. This version uses tabular production and inspection data.

---

## 4. Target Users

### Quality Engineer
- Reviews defect-risk predictions.
- Investigates top contributing factors.
- Uses batch scoring to prioritize lots for inspection.
- Validates model outputs against real production outcomes.

### Production Supervisor
- Uses risk scores to focus attention on high-risk shifts, lines, or process conditions.
- Reviews operational drivers such as temperature, vibration, supplier grade, or maintenance age.

### Data Analyst / Data Scientist
- Uploads labeled datasets.
- Retrains the model.
- Reviews performance metrics and feature importance.
- Extends the model pipeline.

### Plant Manager
- Reviews aggregate risk patterns and quality-improvement recommendations.
- Tracks operational value such as reduced waste, rework, and inspection backlog.

---

## 5. Core User Stories

### US-001: Train Model
As a data analyst, I want to train a classifier from a labeled manufacturing CSV so that the system can classify defect risk using current process data.

**Acceptance Criteria**
- The user can upload a CSV or use the sample dataset.
- The user can specify the target column.
- The backend validates that the target exists and contains at least two classes.
- The backend returns accuracy, weighted precision, weighted recall, weighted F1, classification report, and confusion matrix.
- The trained model artifact is saved locally.

### US-002: Single Prediction
As a quality engineer, I want to enter process values for one product or lot and receive a defect-risk class so that I can prioritize inspection.

**Acceptance Criteria**
- The UI renders inputs dynamically from trained model metadata.
- Numeric inputs use training statistics for default values.
- Categorical inputs use known training categories.
- The prediction response includes predicted risk class, confidence, class probabilities, and explanation factors.

### US-003: Batch Prediction
As a quality engineer, I want to upload a CSV of many products/lots and download scored results so that I can rank inspection workload.

**Acceptance Criteria**
- The user can upload a CSV through Streamlit.
- The backend scores all rows.
- The UI displays predictions and probabilities.
- The user can download the scored CSV.
- Extra columns from the uploaded CSV are preserved.

### US-004: Explain Prediction
As a manufacturing stakeholder, I want to see why the model predicted a risk class so that I can trust and act on the result.

**Acceptance Criteria**
- The prediction response contains top contributing features.
- SHAP TreeExplainer is used when available and compatible.
- If SHAP fails, the system returns global feature importance fallback instead of failing prediction.
- Explanation output identifies the feature, method, contribution, and effect.

### US-005: View Model Insights
As a data scientist, I want to view model metrics and feature importance so that I can evaluate model quality.

**Acceptance Criteria**
- The UI displays accuracy, precision, recall, and F1.
- The UI displays top feature importance as a chart and table.
- The UI displays confusion matrix.
- The UI displays model metadata including trained date, classes, feature columns, and model type.

---

## 6. Functional Requirements

### 6.1 Data Ingestion
- Accept CSV files through the backend `/train` endpoint.
- Accept CSV files through Streamlit upload controls.
- Include a bundled sample dataset for immediate demo execution.
- Preserve extra columns during batch prediction output where possible.
- Reject empty datasets.
- Reject training datasets missing the target column.

### 6.2 Data Cleaning
- Strip whitespace from column names.
- Drop duplicate rows.
- Drop completely empty columns.
- Drop rows where the target label is missing.
- Handle missing numeric values using median imputation.
- Handle missing categorical values using most-frequent imputation.
- Exclude common ID columns from model features, such as `product_id`, `timestamp`, `created_at`, and `updated_at`.

### 6.3 Feature Handling
- Automatically infer numeric and categorical features.
- Scale numeric features using `StandardScaler`.
- Encode categorical features using `OneHotEncoder` with unknown-category handling.
- Store feature list, numeric feature list, categorical feature list, categorical values, and numeric summary statistics in the model artifact.

### 6.4 Model Training
- Train a `RandomForestClassifier` with balanced class weighting.
- Use a train/test split with stratification when class counts allow.
- Compute standard multiclass classification metrics.
- Save the trained preprocessing and model pipeline as a single artifact.
- Allow retraining through API and frontend.

### 6.5 Prediction
- Accept one JSON record through `/predict`.
- Accept JSON batch records through `/predict-json-batch`.
- Accept CSV batch uploads through `/batch-predict`.
- Auto-fill missing feature columns with null values before preprocessing.
- Return predicted risk class, confidence, and class probabilities.

### 6.6 Explainability
- Use SHAP TreeExplainer for local explanation when installed and compatible.
- Aggregate transformed one-hot feature contributions back to raw feature names.
- Return top explanation factors per prediction.
- Fall back to global feature importance when SHAP is unavailable or fails.

### 6.7 Frontend UI
- Provide tabs for Overview, Single Prediction, Batch Prediction, Train Model, Model Insights, and PRD.
- Allow users to configure backend API URL.
- Show backend health status.
- Show sample data.
- Display prediction outputs clearly.
- Display batch scoring table and CSV download button.
- Render the PRD markdown directly inside the app.

### 6.8 API Documentation
- FastAPI automatically exposes interactive documentation at `/docs`.
- The API must include endpoint names, request schemas, and response examples through FastAPI metadata where possible.

---

## 7. Non-Functional Requirements

### Performance
- Single prediction response should complete in under 2 seconds for typical tabular models on a local machine.
- Batch prediction should support at least 5,000 rows in a single CSV for demo use.
- Training on the bundled sample dataset should complete within a few seconds on a normal laptop.

### Reliability
- API should remain available even if no model is loaded.
- If a model is missing, the backend should attempt to train from sample data at startup.
- Prediction should not fail solely because SHAP is unavailable.

### Maintainability
- Backend code should separate API routing, model logic, and data cleaning.
- Frontend code should keep API calls and UI rendering logically separated.
- Configuration should live in one backend config module.

### Security
- Version 1 is intended for local or internal use.
- No authentication is included by default.
- Production deployment must add authentication, authorization, upload-size limits, audit logging, and network controls.
- Uploaded files should be treated as untrusted input.

### Explainability and Trust
- Predictions must include probabilities, not only labels.
- The model card must state intended use and limitations.
- The UI must not frame predictions as guaranteed defect outcomes.

---

## 8. Data Requirements

### Minimum Required Dataset Structure
Each row should represent one inspected product, lot, batch, or time window.

| Column Type | Example Columns | Required | Notes |
|---|---|---:|---|
| Identifier | `product_id`, `timestamp` | No | Excluded from model by default |
| Product context | `product_type`, `supplier_grade` | Recommended | Helps segment risk |
| Production context | `line_id`, `shift`, `operator_experience_months` | Recommended | Captures operational variation |
| Machine parameters | `machine_temperature_c`, `vibration_mm_s`, `pressure_bar` | Recommended | Core process signals |
| Environment | `humidity_pct` | Optional | Useful where materials are humidity-sensitive |
| Inspection readings | `inspection_score`, `surface_roughness_um` | Recommended | Strong indicators of quality state |
| Maintenance context | `maintenance_days_since` | Recommended | Captures equipment condition |
| Target | `defect_risk` | Yes | Classification label, for example Low/Medium/High |

### Sample Target Classes
- `Low`: Low likelihood of defect or quality issue.
- `Medium`: Elevated risk; inspect if capacity allows.
- `High`: High priority for inspection, process review, or containment.

---

## 9. Step-by-Step Workflow

This implementation maps to the project workflow shown in the catalog.

1. Define defect-risk classification objective.
2. Understand manufacturing process flow and define unit of prediction.
3. Collect production and inspection datasets.
4. Merge datasets and handle inconsistent keys.
5. Clean noise and missing records.
6. Perform exploratory feature analysis.
7. Identify defect-related variables.
8. Encode categorical process parameters.
9. Split data into training and testing sets.
10. Train defect-risk classification model.
11. Evaluate precision, recall, F1, accuracy, and confusion matrix.
12. Interpret feature importance and local prediction drivers.
13. Generate defect-risk insights.
14. Save trained model artifact.
15. Develop Streamlit UI for process inputs.
16. Test defect predictions using sample cases.
17. Deploy Streamlit app and FastAPI backend.
18. Document quality-improvement suggestions and limitations.

---

## 10. System Architecture

### Components

#### Backend: FastAPI
Responsibilities:
- Load or train model.
- Expose health, metadata, metrics, training, prediction, batch scoring, sample data, and model-card endpoints.
- Own preprocessing, model inference, artifact persistence, and explainability.

#### Model Pipeline: scikit-learn
Responsibilities:
- Impute missing values.
- Scale numeric features.
- Encode categorical features.
- Train classifier.
- Produce class probabilities.

#### Frontend: Streamlit
Responsibilities:
- Provide user interface for model operation.
- Display metrics and model insights.
- Support single and batch predictions.
- Support model retraining.
- Render this PRD.

#### Storage
Responsibilities:
- Store sample dataset in `data/sample_defects.csv`.
- Store trained model in `artifacts/defect_risk_model.joblib`.

---

## 11. API Specification

### GET `/health`
Returns backend health and model load status.

Example response:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### GET `/metadata`
Returns model metadata, features, classes, metrics, feature importance, and input UI hints.

### GET `/metrics`
Returns model metrics only.

### GET `/sample-data?limit=10`
Returns sample dataset rows as JSON.

### GET `/sample-data.csv?limit=250`
Returns sample dataset rows as CSV.

### POST `/train`
Retrains model. Accepts multipart form data.

Fields:
- `file`: optional CSV upload.
- `target_col`: target column name. Default is `defect_risk`.

### POST `/predict`
Scores one record.

Example request:
```json
{
  "record": {
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
}
```

Example response:
```json
{
  "prediction": "High",
  "confidence": 0.78,
  "probabilities": {
    "High": 0.78,
    "Low": 0.04,
    "Medium": 0.18
  },
  "explanation": [
    {
      "feature": "inspection_score",
      "method": "SHAP TreeExplainer",
      "contribution": 0.11,
      "effect": "increases predicted class"
    }
  ]
}
```

### POST `/predict-json-batch`
Scores multiple JSON records.

### POST `/batch-predict`
Scores uploaded CSV and returns rows with prediction columns.

### GET `/model-card`
Returns intended use, limitations, target, classes, features, metrics, and explainability statement.

---

## 12. Frontend Requirements

### Overview Tab
Must show:
- Product title and short description.
- Accuracy, precision, recall, and F1.
- Model type.
- Target column.
- Training row count.
- Sample data preview.

### Single Prediction Tab
Must show:
- Dynamic input form based on backend metadata.
- Numeric inputs with training means as defaults.
- Categorical dropdowns with known categories.
- Prediction class and confidence.
- Probability chart.
- Explanation table.

### Batch Prediction Tab
Must show:
- CSV upload control.
- Button to score records.
- Prediction table.
- Download scored CSV button.

### Train Model Tab
Must show:
- Optional CSV upload.
- Target column input.
- Train button.
- Training result and metrics.

### Model Insights Tab
Must show:
- Metric cards.
- Feature-importance chart.
- Feature-importance table.
- Confusion matrix.
- Metadata JSON.

### PRD Tab
Must render this markdown document inside the app.

---

## 13. Model Evaluation Plan

### Primary Metrics
- Weighted precision: useful when false positives are costly.
- Weighted recall: useful when missing high-risk defects is costly.
- Weighted F1: balances precision and recall across imbalanced classes.
- Confusion matrix: identifies where High risk is being confused with Medium or Low.

### Minimum Initial Acceptance Thresholds
For the bundled sample dataset:
- Accuracy: at least 0.75.
- Weighted F1: at least 0.75.
- High-risk recall: should be reviewed manually and improved if below operational tolerance.

For real plant data, thresholds must be established with the quality team because false negatives and false positives have different costs by process.

---

## 14. Quality-Improvement Insight Examples

The system should support insights such as:
- High vibration and long maintenance interval are strong risk drivers.
- Night shift has elevated risk under high machine temperature.
- Supplier grade C materials have higher defect-risk probability.
- Lower inspection scores and higher surface roughness increase High-risk classification.
- Certain production lines require targeted calibration or maintenance review.

These insights are not automatic corrective actions. They are decision-support outputs for quality engineers.

---

## 15. Testing Requirements

### Backend Tests
- Health endpoint returns status `ok`.
- Metadata endpoint returns model classes and features.
- Train endpoint succeeds with sample dataset.
- Predict endpoint returns prediction, confidence, probabilities, and explanation.
- Batch endpoint returns same number of rows as uploaded input.
- Missing optional feature values are handled by the pipeline.
- Missing target column during training returns a clear error.

### Frontend Tests
- App loads when backend is available.
- Backend health check works.
- Single prediction form renders all feature fields.
- Batch upload displays scored results.
- Train button retrains and updates metrics.
- PRD tab renders markdown.

### Manual QA Scenarios
1. Start backend without model artifact. Confirm sample model trains automatically.
2. Delete `artifacts/defect_risk_model.joblib`, restart backend, and confirm recovery.
3. Upload a CSV with extra columns and confirm output preserves them.
4. Upload a CSV missing several features and confirm imputation handles missing values.
5. Upload a CSV missing the target column to `/train` and confirm error handling.

---

## 16. Deployment Plan

### Local Development
1. Create Python virtual environment.
2. Install requirements.
3. Start FastAPI backend.
4. Start Streamlit frontend.
5. Train or use auto-trained sample model.

### Internal Demo Deployment
- Run backend on an internal host.
- Run Streamlit on an internal host.
- Set Streamlit backend URL to the backend service URL.
- Restrict network access to internal users.

### Production Hardening Required Before Plant Deployment
- Add authentication and role-based access.
- Add upload-size limits and file validation.
- Add structured logging and audit history.
- Add model versioning.
- Add data drift monitoring.
- Add scheduled retraining workflow.
- Add validation set from recent production data.
- Add rollback for bad model releases.
- Add HTTPS and secrets management.

---

## 17. Project Folder Structure

```text
defect_risk_app/
  backend/
    main.py
    config.py
    schemas.py
    train.py
    services/
      data_service.py
      model_service.py
  frontend/
    app.py
  data/
    sample_defects.csv
  docs/
    PRD.md
  artifacts/
    defect_risk_model.joblib
  scripts/
    generate_sample_data.py
  tests/
    test_backend_smoke.py
  requirements.txt
  README.md
  docker-compose.yml
```

---

## 18. Execution Instructions

### Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Start Backend
```bash
uvicorn backend.main:app --reload --port 8000
```

### Start Frontend
Open a second terminal:
```bash
streamlit run frontend/app.py
```

### Train from CLI
```bash
python -m backend.train data/sample_defects.csv --target defect_risk
```

### Run Tests
```bash
pytest -q
```

---

## 19. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Poor defect labels | Bad model learning | Validate labels, remove noisy periods, audit sampling |
| Class imbalance | Model misses High-risk cases | Use class weighting, inspect class-specific recall, resample if needed |
| Process drift | Model becomes stale | Monitor recent performance and retrain regularly |
| Unknown categories | Prediction errors | Use one-hot encoder with unknown handling |
| Black-box distrust | Low adoption | Provide probabilities, explanations, model card, and limitations |
| Over-automation | Unsafe operational decisions | Keep human review in loop; do not auto-reject without validation |

---

## 20. Future Enhancements

- Add user authentication.
- Add model registry and version comparison.
- Add drift monitoring dashboard.
- Add scheduled retraining.
- Add database persistence for predictions and feedback.
- Add feedback loop where engineers mark whether prediction was correct.
- Add cost-sensitive threshold tuning.
- Add per-line or per-product specialized models.
- Add image-based defect detection as a separate computer vision module.
- Add integration with MES, ERP, or quality management systems.

---

## 21. Definition of Done

The version is complete when:
- Backend starts successfully.
- Streamlit app connects to backend.
- Sample model trains automatically or through UI.
- Single prediction works.
- Batch prediction works.
- Metrics and feature importance display.
- PRD renders in frontend.
- README provides local execution commands.
- Project can be zipped and shared as a runnable deliverable.
