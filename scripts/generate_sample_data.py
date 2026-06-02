import numpy as np
import pandas as pd
import os

def generate_data(filepath, n_rows=5000):
    print(f"Generating synthetic manufacturing defect risk dataset at {filepath}...")
    np.random.seed(42)

    product_types = ['motor', 'gear', 'valve', 'assembly']
    line_ids = ['line_1', 'line_2', 'line_3']
    shifts = ['morning', 'evening', 'night']
    supplier_grades = ['A', 'B', 'C']
    inspection_methods = ['camera', 'ultrasound', 'visual', 'manual']

    # Generate random features based on physical ranges
    data = {
        'product_id': [f"PRD-{i:05d}" for i in range(1, n_rows + 1)],
        'timestamp': pd.date_range(start='2026-01-01', periods=n_rows, freq='h').strftime('%Y-%m-%d %H:%M:%S'),
        'product_type': np.random.choice(product_types, size=n_rows, p=[0.4, 0.3, 0.2, 0.1]),
        'line_id': np.random.choice(line_ids, size=n_rows, p=[0.5, 0.3, 0.2]),
        'shift': np.random.choice(shifts, size=n_rows, p=[0.4, 0.4, 0.2]),
        'supplier_grade': np.random.choice(supplier_grades, size=n_rows, p=[0.3, 0.5, 0.2]),
        'inspection_method': np.random.choice(inspection_methods, size=n_rows, p=[0.3, 0.2, 0.4, 0.1]),
        'machine_temperature_c': np.random.normal(85.0, 8.0, n_rows),
        'vibration_mm_s': np.random.normal(3.5, 1.2, n_rows),
        'humidity_pct': np.random.normal(55.0, 8.0, n_rows),
        'pressure_bar': np.random.normal(6.0, 1.2, n_rows),
        'conveyor_speed_m_min': np.random.normal(30.0, 4.0, n_rows),
        'material_batch_age_days': np.random.randint(5, 90, n_rows),
        'operator_experience_months': np.random.randint(3, 120, n_rows),
        'maintenance_days_since': np.random.randint(1, 90, n_rows),
        'inspection_score': np.random.normal(82.0, 7.5, n_rows),
        'surface_roughness_um': np.random.normal(2.5, 0.8, n_rows),
    }

    df = pd.DataFrame(data)
    
    # Clip bounds for sanity
    df['inspection_score'] = df['inspection_score'].clip(30.0, 100.0)
    df['surface_roughness_um'] = df['surface_roughness_um'].clip(0.1, 10.0)

    # Apply physical rules to establish correlation and model explainability (PRD Section 14)
    risk_score = np.zeros(n_rows)

    # 1. High vibration and long maintenance interval -> High Risk
    risk_score += np.where((df['vibration_mm_s'] > 4.8) & (df['maintenance_days_since'] > 60), 0.35, 0.0)
    
    # 2. Night shift and high machine temperature -> High Risk
    risk_score += np.where((df['shift'] == 'night') & (df['machine_temperature_c'] > 94.0), 0.3, 0.0)
    
    # 3. Supplier grade C -> higher risk
    risk_score += np.where(df['supplier_grade'] == 'C', 0.18, 0.0)
    
    # 4. Low operator experience -> slight risk increase
    risk_score += np.where(df['operator_experience_months'] < 12, 0.1, 0.0)
    
    # 5. Conveyor speed too high -> slight increase
    risk_score += np.where(df['conveyor_speed_m_min'] > 36.5, 0.15, 0.0)
    
    # 6. Inspection score and roughness -> strong indicators
    risk_score += np.where(df['inspection_score'] < 72.0, 0.45, 0.0)
    risk_score += np.where(df['surface_roughness_um'] > 3.8, 0.25, 0.0)

    # Add Gaussian noise to make it realistic
    noise = np.random.normal(0, 0.08, n_rows)
    total_score = risk_score + noise

    # Map scores to the target High/Medium/Low classes
    defect_risk = []
    for s in total_score:
        if s >= 0.45:
            defect_risk.append('High')
        elif s >= 0.18:
            defect_risk.append('Medium')
        else:
            defect_risk.append('Low')
            
    df['defect_risk'] = defect_risk

    # Create directories if not exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Successfully generated {n_rows} rows. Defect risk distribution:")
    print(df['defect_risk'].value_counts())

if __name__ == "__main__":
    generate_data("data/sample_defects.csv")
