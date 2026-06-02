import argparse
import pandas as pd
from backend.services.model_service import ModelService
from backend.config import DEFAULT_TARGET_COL

def main():
    parser = argparse.ArgumentParser(description="Train Defect Risk Classification Model")
    parser.add_argument("data_path", type=str, help="Path to labeled training CSV file")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET_COL, help="Target column name")
    
    args = parser.parse_args()
    
    print(f"Loading training data from {args.data_path}...")
    try:
        df = pd.read_csv(args.data_path)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    service = ModelService()
    print("Initializing pipeline training and cross-validation...")
    try:
        metadata = service.train_pipeline(df, target_col=args.target)
        print("\n" + "="*40)
        print("Model Retraining Successful")
        print(f"  Trained Date: {metadata['trained_at']}")
        print(f"  Sample Size:  {metadata['metrics']['sample_size']} rows")
        print(f"  Accuracy:     {metadata['metrics']['accuracy']:.4f}")
        print(f"  Precision:    {metadata['metrics']['precision']:.4f}")
        print(f"  Recall:       {metadata['metrics']['recall']:.4f}")
        print(f"  F1-Score:     {metadata['metrics']['f1']:.4f}")
        print("="*40)
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()
