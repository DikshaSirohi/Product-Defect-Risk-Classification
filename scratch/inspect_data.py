import pandas as pd
import os

def inspect_file(filepath):
    print(f"=== Inspecting {os.path.basename(filepath)} ===")
    if not os.path.exists(filepath):
        print("File does not exist.")
        return
    df = pd.read_csv(filepath)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nData Types:")
    print(df.dtypes)
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nNull counts:")
    print(df.isnull().sum())
    
    # Check target distributions
    if 'DefectStatus' in df.columns:
        print("\nTarget 'DefectStatus' distribution:")
        print(df['DefectStatus'].value_counts())
    if 'Machine failure' in df.columns:
        print("\nTarget 'Machine failure' distribution:")
        print(df['Machine failure'].value_counts())
    print("\n" + "="*50 + "\n")

inspect_file("clean_ai4i2020.csv")
inspect_file("cleaned_manufacturing_defects_dataset.csv")
