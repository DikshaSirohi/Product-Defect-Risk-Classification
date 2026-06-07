import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Ensure root directory is in python path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from backend.services.model_service import ModelService
from backend.config import SAMPLE_DATA_PATH, DEFAULT_TARGET_COL

# Set Premium Page Configurations
st.set_page_config(
    page_title="AI Defect Risk Control Center",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium HSL Styling and Animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3c72;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Badges */
    .badge {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-low {
        background-color: hsl(120, 60%, 85%);
        color: hsl(120, 60%, 25%);
        border: 1px solid hsl(120, 60%, 70%);
    }
    .badge-medium {
        background-color: hsl(45, 90%, 85%);
        color: hsl(45, 90%, 25%);
        border: 1px solid hsl(45, 90%, 70%);
    }
    .badge-high {
        background-color: hsl(0, 75%, 85%);
        color: hsl(0, 75%, 25%);
        border: 1px solid hsl(0, 75%, 70%);
    }

    /* Process Flow Styles */
    .flow-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 2rem 0;
    }
    .flow-step {
        background: white;
        border-left: 5px solid #1e3c72;
        padding: 1rem 1.5rem;
        margin: 0.4rem 0;
        width: 80%;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        display: flex;
        align-items: center;
    }
    .flow-step-num {
        background: #1e3c72;
        color: white;
        border-radius: 50%;
        width: 32px;
        height: 32px;
        display: flex;
        justify-content: center;
        align-items: center;
        font-weight: bold;
        margin-right: 1.5rem;
        flex-shrink: 0;
    }
    .flow-step-details h4 {
        margin: 0;
        color: #333;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .flow-step-details p {
        margin: 0.2rem 0 0 0;
        color: #777;
        font-size: 0.85rem;
    }
    .flow-arrow {
        color: #1e3c72;
        font-size: 1.5rem;
        margin: 0.2rem 0;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Initialize Model Service -----------------
if "model_service" not in st.session_state:
    with st.spinner("Initializing Model Service and loading model..."):
        service = ModelService()
        loaded = service.load_model()
        if not loaded:
            # Try to auto-train if model is not loaded
            if os.path.exists(SAMPLE_DATA_PATH):
                try:
                    df = pd.read_csv(SAMPLE_DATA_PATH)
                    service.train_pipeline(df, DEFAULT_TARGET_COL)
                except Exception as e:
                    st.sidebar.error(f"Failed to auto-train model: {e}")
            else:
                st.sidebar.error(f"Sample data missing at {SAMPLE_DATA_PATH}")
        st.session_state.model_service = service

model_service = st.session_state.model_service
connection_ok = True
model_loaded = model_service.is_model_loaded()

# ----------------- SIDEBAR: Control Settings & Connection Status -----------------
logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logo.jpg")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=70)
else:
    # Try direct relative path
    if os.path.exists("logo.jpg"):
        st.sidebar.image("logo.jpg", width=70)
    else:
        st.sidebar.image("https://img.icons8.com/color/96/automation-precision.png", width=70)
st.sidebar.markdown("### Control Settings")

# Display Connection light
if model_loaded:
    st.sidebar.success("🟢 Active | Model Loaded")
else:
    st.sidebar.error("🔴 Active | Model Missing")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small><b>Intended Use</b>: Demo purposes.</small>", 
    unsafe_allow_html=True
)

# ----------------- MAIN TITLE HEADER -----------------
st.markdown("""
<div class="main-header">
    <h1>AI Defect Risk Control Center</h1>
    <p>Predictive Quality Decision Support & Process Optimizations</p>
</div>
""", unsafe_allow_html=True)

# ----------------- TABS SYSTEM -----------------
tabs = st.tabs([
    "📈 Overview", 
    "🎯 Single Prediction", 
    "📂 Batch Prediction", 
    "🔄 Train Model", 
    "🔬 Model Insights"
])

# ----------------- TAB 1: OVERVIEW -----------------
with tabs[0]:
    st.markdown("### Process Overview & Model Status")
    st.markdown(
        "**Developer Perspective & Model Architecture**:\n"
        "This application implements a high-performance **HistGradientBoostingClassifier** to monitor, diagnose, and predict defect risk "
        "across a 10-step assembly process. By integrating data across raw material quality, tool calibration parameters, machine telemetry "
        "(vibration, temperature, pressure), and in-process inspection scores, the machine learning pipeline extracts complex non-linear correlations "
        "that traditional rule-based threshold systems miss.\n\n"
        "The system computes local feature explanations using **SHAP (SHapley Additive exPlanations)**, giving quality control engineers "
        "full visibility into the specific drivers shifting the risk profile of each assembly. When SHAP calculations face numerical edge cases, "
        "the backend automatically falls back to **Permutation Feature Importance** to maintain zero-downtime reliability."
    )
    
    # Render Model Metrics Cards
    if connection_ok and model_loaded:
        try:
            metrics = model_service.get_metadata().get("metrics", {})
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{metrics.get('accuracy', 0.0)*100:.2f}%</div><div class='metric-label'>Accuracy</div></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{metrics.get('precision', 0.0)*100:.2f}%</div><div class='metric-label'>Precision (Weighted)</div></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{metrics.get('recall', 0.0)*100:.2f}%</div><div class='metric-label'>Recall (Weighted)</div></div>", unsafe_allow_html=True)
            with col4:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{metrics.get('f1', 0.0)*100:.2f}%</div><div class='metric-label'>F1-Score (Weighted)</div></div>", unsafe_allow_html=True)
        except Exception:
            st.info("Failed to pull real-time model metrics.")
    else:
        st.warning("Please train a model to view status.")

    st.markdown("---")
    
    # Split layout for Process Flow and Sample Data
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.markdown("### ⚙️ Interactive Manufacturing Process Flow")
        st.markdown(
            "Below is the standard 10-step assembly process. Our AI monitors variables "
            "across key calibration phases to predict down-stream defect risks."
        )
        
        # HTML visual process flow matching the user's defined flow and PRD
        st.markdown("""
        <div class="flow-container">
            <div class="flow-step">
                <div class="flow-step-num">1</div>
                <div class="flow-step-details">
                    <h4>Raw Material Inspection</h4>
                    <p>Monitors <strong>Supplier Quality</strong> and grade to establish baseline materials safety.</p>
                </div>
            </div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step">
                <div class="flow-step-num">2</div>
                <div class="flow-step-details">
                    <h4>Material Preparation</h4>
                    <p>Prepares additive material stocks, capturing <strong>Additive Process Time</strong> & Cost.</p>
                </div>
            </div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step">
                <div class="flow-step-num">3</div>
                <div class="flow-step-details">
                    <h4>Machine Setup & Calibration</h4>
                    <p>Calibrates tool state and notes <strong>Days since maintenance</strong> to prevent drift.</p>
                </div>
            </div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step">
                <div class="flow-step-num">4</div>
                <div class="flow-step-details">
                    <h4>Production / Assembly Process</h4>
                    <p>Active tooling. Logs <strong>temperature</strong>, <strong>vibration</strong>, pressure, speed, and shift context.</p>
                </div>
            </div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step">
                <div class="flow-step-num">5</div>
                <div class="flow-step-details">
                    <h4>In-Process Quality Inspection</h4>
                    <p>In-line testing of surface characteristics (e.g. <strong>surface roughness</strong>).</p>
                </div>
            </div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step">
                <div class="flow-step-num">6</div>
                <div class="flow-step-details">
                    <h4>Defect Detection & Recording</h4>
                    <p>Real-time machine learning predictions of <strong>Defect Risk</strong> (Low, Medium, High).</p>
                </div>
            </div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step">
                <div class="flow-step-num">7</div>
                <div class="flow-step-details">
                    <h4>Rework or Scrap Handling</h4>
                    <p>Automated screening isolates high-risk lots, reducing scrap and waste.</p>
                </div>
            </div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step">
                <div class="flow-step-num">8</div>
                <div class="flow-step-details">
                    <h4>Final Quality Testing</h4>
                    <p>Determines overall <strong>Quality Score</strong> and certifies components for shipping.</p>
                </div>
            </div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step">
                <div class="flow-step-num">9</div>
                <div class="flow-step-details">
                    <h4>Packaging</h4>
                    <p>Components packaged securely in line with shipping specifications.</p>
                </div>
            </div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step">
                <div class="flow-step-num">10</div>
                <div class="flow-step-details">
                    <h4>Storage / Dispatch</h4>
                    <p>Stored or shipped to destination. Logs delivery delays and logistical turnover.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        st.markdown("### 📊 Active Training Sample Data")
        st.markdown(
            "Below is a preview of the default dataset rows used to train and calibrate the classifier. "
            "Click download to pull a copy for local analysis."
        )
        
        if connection_ok:
            try:
                # Load sample data
                if os.path.exists(SAMPLE_DATA_PATH):
                    df_sample_full = pd.read_csv(SAMPLE_DATA_PATH)
                    df_sample = df_sample_full.head(15)
                    st.dataframe(df_sample, height=450)
                    
                    # Native Streamlit Download Button
                    csv_content = df_sample_full.head(500).to_csv(index=False)
                    st.download_button(
                        label="📥 Download 500 Sample Rows (CSV)",
                        data=csv_content,
                        file_name="sample_defects_500.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Sample dataset file not found.")
            except Exception as e:
                st.error(f"Error loading sample data: {e}")
        else:
            st.info("Offline: Connect backend to inspect training data.")

# ----------------- TAB 2: SINGLE PREDICTION -----------------
with tabs[1]:
    st.markdown("### 🎯 Single-Record Defect Risk Predictor")
    st.markdown(
        "**Developer Perspective & Real-Time Scoring**:\n"
        "This interface allows operators and quality engineers to perform real-time risk assessment for a specific lot or tooling window. "
        "The backend dynamically inspects the loaded model's metadata to render the inputs, pre-filling numerical inputs with historical "
        "medians and categorical selections with known training classes to avoid out-of-distribution inputs.\n\n"
        "Upon submission, the data is preprocessed through a scikit-learn pipeline (incorporating median imputation for missing fields, "
        "StandardScaler for numeric scaling, and OneHotEncoder for categoricals). The backend then calculates class probability distributions "
        "and returns **local SHAP explainability drivers** to indicate which physical factors (such as excessive vibration or surface roughness) "
        "are actively pushing the risk boundary."
    )
    
    if connection_ok and model_loaded:
        try:
            # Dynamically fetch training metadata for default pre-fills
            metadata = model_service.get_metadata()
            features = metadata.get("features", [])
            numeric_features = metadata.get("numeric_features", [])
            categorical_features = metadata.get("categorical_features", [])
            defaults = metadata.get("feature_defaults", {})
            categories = metadata.get("categorical_categories", {})

            # Build Form dynamically pre-filled with statistical training means/modes
            with st.form("predict_form"):
                st.markdown("#### 1. Material & Tool Setup Context")
                c1, c2, c3 = st.columns(3)
                
                form_inputs = {}
                
                # Check categorical columns
                if "product_type" in categorical_features:
                    with c1:
                        form_inputs["product_type"] = st.selectbox(
                            "Product Type", 
                            categories.get("product_type", ["motor", "gear", "valve", "assembly"]),
                            index=0
                        )
                if "supplier_grade" in categorical_features:
                    with c2:
                        form_inputs["supplier_grade"] = st.selectbox(
                            "Supplier Grade", 
                            categories.get("supplier_grade", ["A", "B", "C"]),
                            index=1 if len(categories.get("supplier_grade", [])) > 1 else 0
                        )
                if "line_id" in categorical_features:
                    with c3:
                        form_inputs["line_id"] = st.selectbox(
                            "Production Line ID", 
                            categories.get("line_id", ["line_1", "line_2", "line_3"]),
                            index=0
                        )

                st.markdown("#### 2. Assembly & Process Settings")
                c4, c5, c6, c7 = st.columns(4)
                
                if "shift" in categorical_features:
                    with c4:
                        form_inputs["shift"] = st.selectbox(
                            "Working Shift", 
                            categories.get("shift", ["morning", "evening", "night"]),
                            index=0
                        )
                if "machine_temperature_c" in numeric_features:
                    with c5:
                        val = defaults.get("machine_temperature_c", 85.0)
                        form_inputs["machine_temperature_c"] = st.slider("Machine Temperature (°C)", 40.0, 130.0, float(val))
                if "vibration_mm_s" in numeric_features:
                    with c6:
                        val = defaults.get("vibration_mm_s", 3.5)
                        form_inputs["vibration_mm_s"] = st.slider("Vibration (mm/s)", 0.0, 10.0, float(val))
                if "pressure_bar" in numeric_features:
                    with c7:
                        val = defaults.get("pressure_bar", 6.0)
                        form_inputs["pressure_bar"] = st.slider("Process Pressure (bar)", 1.0, 15.0, float(val))

                st.markdown("#### 3. Maintenance & Experience Indicators")
                c8, c9, c10, c11 = st.columns(4)
                
                if "conveyor_speed_m_min" in numeric_features:
                    with c8:
                        val = defaults.get("conveyor_speed_m_min", 30.0)
                        form_inputs["conveyor_speed_m_min"] = st.slider("Conveyor Speed (m/min)", 5.0, 60.0, float(val))
                if "operator_experience_months" in numeric_features:
                    with c9:
                        val = defaults.get("operator_experience_months", 24.0)
                        form_inputs["operator_experience_months"] = st.number_input("Operator Experience (months)", 0, 500, int(val))
                if "maintenance_days_since" in numeric_features:
                    with c10:
                        val = defaults.get("maintenance_days_since", 30.0)
                        form_inputs["maintenance_days_since"] = st.number_input("Days Since Maintenance", 0, 365, int(val))
                if "material_batch_age_days" in numeric_features:
                    with c11:
                        val = defaults.get("material_batch_age_days", 45.0)
                        form_inputs["material_batch_age_days"] = st.number_input("Material Batch Age (days)", 0, 365, int(val))

                st.markdown("#### 4. Quality Readings")
                c12, c13, c14 = st.columns(3)
                
                if "inspection_score" in numeric_features:
                    with c12:
                        val = defaults.get("inspection_score", 82.0)
                        form_inputs["inspection_score"] = st.slider("In-Process Inspection Score", 20.0, 100.0, float(val))
                if "surface_roughness_um" in numeric_features:
                    with c13:
                        val = defaults.get("surface_roughness_um", 2.5)
                        form_inputs["surface_roughness_um"] = st.slider("Surface Roughness (μm)", 0.0, 10.0, float(val))
                if "inspection_method" in categorical_features:
                    with c14:
                        form_inputs["inspection_method"] = st.selectbox(
                            "Inspection Method", 
                            categories.get("inspection_method", ["camera", "ultrasound", "visual", "manual"]),
                            index=0
                        )

                # Collect fallback fields that might be in training dataset but not in form
                for f_name in features:
                    if f_name not in form_inputs:
                        form_inputs[f_name] = defaults.get(f_name, 0.0)

                submitted = st.form_submit_button("🔍 Run Defect Risk Assessment")
            
            # Predict Trigger
            if submitted:
                with st.spinner("Classifying and explaining risk profile..."):
                    pred_res = model_service.predict_record(form_inputs)
                    
                # Layout results
                col_res_left, col_res_right = st.columns([1, 1.2])
                
                with col_res_left:
                    st.markdown("### Prediction Classification Output")
                    pred_class = pred_res.get("prediction")
                    confidence = pred_res.get("confidence", 0.0)
                    
                    # Highlight prediction banner
                    badge_style = "badge-low"
                    if pred_class == "Medium":
                        badge_style = "badge-medium"
                    elif pred_class == "High":
                        badge_style = "badge-high"
                        
                    st.markdown(f"Predicted Defect Risk: <span class='badge {badge_style}'>{pred_class}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Model Confidence**: `{confidence*100:.2f}%`")
                    
                    # Probability Bar Chart
                    st.markdown("#### Class Probability Distribution")
                    probs = pred_res.get("probabilities", {})
                    
                    # Create plot
                    fig, ax = plt.subplots(figsize=(6, 2.5))
                    classes_list = list(probs.keys())
                    prob_vals = [p * 100 for p in probs.values()]
                    
                    colors = ['hsl(120, 60%, 55%)', 'hsl(45, 90%, 55%)', 'hsl(0, 75%, 55%)']
                    # Match colors to index
                    color_mapped = []
                    for c in classes_list:
                        if c == "Low":
                            color_mapped.append("green")
                        elif c == "Medium":
                            color_mapped.append("orange")
                        else:
                            color_mapped.append("red")
                            
                    bars = ax.barh(classes_list, prob_vals, color=color_mapped, height=0.5)
                    ax.set_xlim(0, 100)
                    ax.set_xlabel("Probability (%)")
                    sns.despine(left=True, bottom=True)
                    
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 2, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                                va='center', ha='left', fontweight='bold')
                    
                    st.pyplot(fig)

                with col_res_right:
                    st.markdown("### 🔬 Local Explainability drivers (SHAP)")
                    st.markdown("The most influential parameters shifting this record's predicted defect risk profile:")
                    
                    explanations = pred_res.get("explanation", [])
                    if explanations:
                        df_explain = pd.DataFrame(explanations)
                        # Pretty format table
                        df_explain.columns = ["Tooling Parameter", "Method", "SHAP Impact Score", "Process Direction"]
                        st.table(df_explain)
                    else:
                        st.info("No explainability parameters returned.")
                    
                    # Operational Quality engineering recommendations
                    st.markdown("#### 💡 Quality Improvement Action Guide")
                    if pred_class == "High":
                        st.error(
                            "⚠️ **Action Required**: Isolate the current lot immediately. Perform manual downstream "
                            "quality tests and check calibration settings (e.g. check vibration dampener and tool wear)."
                        )
                    elif pred_class == "Medium":
                        st.warning(
                            "💡 **Inspection Recommended**: Flag this lot for inspection if queue capacity allows. "
                            "Monitor machine temperature trends on subsequent runs."
                        )
                    else:
                        st.success(
                            "✅ **Proceed with Production**: Settings are well within normal bounds. "
                            "Ensure regular batch logistics continue without modification."
                        )

        except Exception as e:
            st.error(f"Error communicating with backend metadata: {e}")
    else:
        st.warning("Please ensure the FastAPI backend is running locally to render input features.")

# ----------------- TAB 3: BATCH PREDICTION -----------------
with tabs[2]:
    st.markdown("### 📂 CSV Batch Predictor")
    st.markdown(
        "**Developer Perspective & Batch Scoring Workflow**:\n"
        "For offline analysis and shift-change reviews, the batch scoring endpoint accepts a CSV upload and processes rows sequentially. "
        "The algorithm handles missing or corrupted telemetry values using the imputation defaults established during training, "
        "ensuring high-availability prediction. \n\n"
        "Crucially, **all metadata and non-modeled tracking columns (e.g., custom timestamps, product IDs, or supplier tags) are preserved** "
        "in the output file. This allows downstream systems (such as MES, ERP, or BI tools) to seamlessly join the AI risk scores back to "
        "material supply chain records."
    )
    
    uploaded_file = st.file_uploader("Upload labeled or unlabeled CSV", type=["csv"])
    
    if uploaded_file is not None:
        if connection_ok and model_loaded:
            # Display file info
            df_input = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded CSV containing {len(df_input)} rows.")
            st.dataframe(df_input.head(5))
            
            # Button to score
            if st.button("🚀 Score Batch Records"):
                with st.spinner("Scoring records in the background..."):
                    try:
                        # Reset file pointer and upload
                        uploaded_file.seek(0)
                        files = {"file": (uploaded_file.name, uploaded_file.read(), "text/csv")}
                        
                        predictions = []
                        confidences = []
                        for _, row in df_input.iterrows():
                            record = row.to_dict()
                            try:
                                pred_res = model_service.predict_record(record)
                                predictions.append(pred_res["prediction"])
                                confidences.append(pred_res["confidence"])
                            except Exception:
                                predictions.append("Error")
                                confidences.append(0.0)
                        
                        df_scored = df_input.copy()
                        df_scored["predicted_defect_risk"] = predictions
                        df_scored["prediction_confidence"] = confidences

                        st.success("Successfully scored all batch rows!")
                        
                        # Render stats
                        total_rows = len(df_scored)
                        counts = df_scored["predicted_defect_risk"].value_counts()
                        
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            low_cnt = counts.get("Low", 0)
                            st.metric("Low Risk Lots", f"{low_cnt} / {total_rows}", f"{low_cnt/total_rows*100:.1f}%")
                        with c2:
                            med_cnt = counts.get("Medium", 0)
                            st.metric("Medium Risk Lots", f"{med_cnt} / {total_rows}", f"{med_cnt/total_rows*100:.1f}%")
                        with c3:
                            high_cnt = counts.get("High", 0)
                            st.metric("High Risk Lots", f"{high_cnt} / {total_rows}", f"{high_cnt/total_rows*100:.1f}%")

                        # Scored Dataframe Preview
                        st.markdown("#### Scored Dataset Preview")
                        st.dataframe(df_scored.head(10))
                        
                        # Download button
                        scored_csv = df_scored.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Scored Batch CSV",
                            data=scored_csv,
                            file_name="scored_defect_risk_results.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Failed to process batch data: {e}")
        else:
            st.warning("Please train a model to enable batch predictions.")

# ----------------- TAB 4: TRAIN MODEL -----------------
with tabs[3]:
    st.markdown("### 🔄 Model Calibration & Retraining Control Panel")
    st.markdown(
        "**Developer Perspective & Pipeline Retraining**:\n"
        "Continuous retraining is vital to mitigate **covariate shift** and **model degradation** over time. "
        "This control panel triggers an automated training workflow on the backend: \n"
        "1. **Ingestion & Validation**: Strips whitespace from columns, drops duplicate rows, and filters out missing target rows.\n"
        "2. **Feature Separation**: Separates features from targets and drops identifiers (e.g., product IDs) using a configuration-driven exclusion list.\n"
        "3. **Pipeline Construction**: Fits median/mode imputers, scales numeric features, and one-hot encodes categorical values.\n"
        "4. **Training & Stratification**: Performs a stratified 80/20 train/test split to maintain class balance before fitting the **HistGradientBoostingClassifier**.\n"
        "5. **Diagnostics**: Calculates weighted precision, recall, and F1-score, generates a confusion matrix, computes permutation feature importances, and serializes the complete pipeline package to disk for zero-latency loading."
    )
    
    # Show persistent success notifications from session state
    if "train_success" in st.session_state:
        st.success(st.session_state.train_success)
        if "train_res" in st.session_state:
            with st.expander("Show Retraining Diagnostics Details"):
                st.json(st.session_state.train_res)
        del st.session_state.train_success
        if "train_res" in st.session_state:
            del st.session_state.train_res

    st.markdown("#### 1. Dataset Configuration")
    dataset_option = st.selectbox(
        "Choose Training Dataset Source",
        [
            "(Recommended) Premium Synthetic Dataset (data/sample_defects.csv)",
            "Workspace Dataset: cleaned_manufacturing_defects_dataset.csv (DefectStatus)",
            "Workspace Dataset: clean_ai4i2020.csv (Machine failure)",
            "Upload custom CSV file..."
        ]
    )

    custom_train_file = None
    target_column_name = "defect_risk"

    # Display dynamic options
    if "sample_defects.csv" in dataset_option:
        target_column_name = "defect_risk"
        st.info("Using standard, pre-calibrated synthetic dataset to achieve optimal (>90%) accuracy.")
    elif "cleaned_manufacturing_defects_dataset.csv" in dataset_option:
        target_column_name = "DefectStatus"
        st.warning("Training on 'cleaned_manufacturing_defects_dataset.csv' using target column 'DefectStatus'.")
    elif "clean_ai4i2020.csv" in dataset_option:
        target_column_name = "Machine failure"
        st.warning("Training on 'clean_ai4i2020.csv' using target column 'Machine failure'.")
    else:
        custom_train_file = st.file_uploader("Upload custom training CSV file", type=["csv"])
        target_column_name = st.text_input("Enter Target Column Name", value="defect_risk")

    # Retrain button
    if st.button("🚀 Trigger Model Re-Training Pipeline"):
        if connection_ok:
            with st.spinner("Retraining classifier pipeline. Preprocessing data, splitting classes, and saving artifacts..."):
                try:
                    if custom_train_file is not None:
                        custom_train_file.seek(0)
                        df = pd.read_csv(custom_train_file)
                    else:
                        local_path = None
                        if "cleaned_manufacturing_defects_dataset.csv" in dataset_option:
                            local_path = "cleaned_manufacturing_defects_dataset.csv"
                        elif "clean_ai4i2020.csv" in dataset_option:
                            local_path = "clean_ai4i2020.csv"
                        
                        if local_path and os.path.exists(local_path):
                            df = pd.read_csv(local_path)
                        else:
                            df = pd.read_csv(SAMPLE_DATA_PATH)
                    
                    metadata = model_service.train_pipeline(df, target_column_name)
                    
                    st.session_state.train_success = "★ Model retraining successful and serialized successfully! ★"
                    st.session_state.train_res = {
                        "status": "success",
                        "message": "Model trained and serialized successfully.",
                        "metrics": metadata.get("metrics"),
                        "trained_at": metadata.get("trained_at")
                    }
                    st.rerun()  # Refresh screen to update metrics
                except Exception as e:
                    st.error(f"Training failed: {e}")
        else:
            st.warning("Model service configuration error.")

# ----------------- TAB 5: MODEL INSIGHTS -----------------
with tabs[4]:
    st.markdown("### 🔬 In-depth Model Analytics & Performance Diagnostics")
    st.markdown(
        "**Developer Perspective & Model Interpretation**:\n"
        "Validating model quality requires inspecting performance across multiple dimensions:\n"
        "- **Confusion Matrix Heatmap**: Shows class-by-class accuracy. Quality engineers can verify that high-risk cases are not being misclassified as low-risk (false negatives), which is critical for waste containment.\n"
        "- **Permutation Feature Importance**: A model-agnostic metric that measures the increase in prediction error when a feature's values are shuffled. Unlike standard Random Forest importances which bias towards high-cardinality numeric features, permutation importance shows the true impact of variables on the model's actual test performance."
    )
    
    if connection_ok and model_loaded:
        try:
            metadata = model_service.get_metadata()
            metrics = metadata.get("metrics", {})
            trained_at = metadata.get("trained_at", "N/A")
            classes = metadata.get("classes", [])
            
            st.markdown(f"**Trained Date**: `{trained_at}` | **Target Variable**: `{metadata.get('target_col', 'defect_risk')}`")
            
            # Sub Columns
            col_insight_left, col_insight_right = st.columns([1, 1.2])
            
            with col_insight_left:
                st.markdown("#### 📊 Confusion Matrix Heatmap")
                st.markdown("Identifies where class classifications mismatch or overlap.")
                
                cm_data = metrics.get("confusion_matrix", {})
                if cm_data and "values" in cm_data:
                    cm_values = cm_data["values"]
                    cm_classes = cm_data.get("classes", classes)
                    
                    # Seaborn plot
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(
                        cm_values, 
                        annot=True, 
                        fmt="d", 
                        cmap="Blues", 
                        xticklabels=cm_classes, 
                        yticklabels=cm_classes, 
                        ax=ax,
                        cbar=False
                    )
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    st.pyplot(fig)
                else:
                    st.info("No confusion matrix available in model metrics.")

            with col_insight_right:
                st.markdown("#### 🌟 Global Feature Importance Chart")
                st.markdown("Displays the relative weight each parameter holds based on permutation importance on the test dataset.")
                
                # Render metadata JSON in expander
                with st.expander("🔍 View Raw Model Metadata JSON"):
                    st.json(metadata)

            # Global importance plotting (using Permutation importances returned by metadata)
            try:
                global_imps = metadata.get("global_importances", {})
                if global_imps:
                    features_sample = list(global_imps.keys())
                    importances_sample = list(global_imps.values())
                    
                    # Take top 10 for plotting
                    features_sample = features_sample[:10]
                    importances_sample = importances_sample[:10]
                    
                    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
                    sns.barplot(x=importances_sample, y=features_sample, palette="viridis", ax=ax2)
                    ax2.set_xlabel("Permutation Importance (Mean Score Decrease)")
                    sns.despine(left=True, bottom=True)
                    st.pyplot(fig2)
                    
                    # Display table
                    df_imp = pd.DataFrame({
                        "Feature": features_sample,
                        "Importance Score": importances_sample
                    })
                    st.table(df_imp)
                else:
                    st.info("No global feature importances available in model metadata. Retrain the model to compute them.")
            except Exception as plot_err:
                st.error(f"Failed to plot importance charts: {plot_err}")

        except Exception as e:
            st.error(f"Failed to fetch model metrics: {e}")
    else:
        st.info("Model missing: Train a model to load insights.")
