import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from backend.config import EXCLUDE_COLUMNS

class DataService:
    @staticmethod
    def clean_data(df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Cleans the input DataFrame according to the PRD specifications.
        """
        if df.empty:
            raise ValueError("Input dataset is empty.")

        # Copy to avoid modifying original
        df = df.copy()

        # Strip whitespace from column names
        df.columns = df.columns.astype(str).str.strip()
        
        # Drop completely empty columns
        df = df.dropna(how='all', axis=1)
        
        # Drop duplicate rows
        df = df.drop_duplicates()
        
        # Drop rows where target label is missing
        if target_col and target_col in df.columns:
            df = df.dropna(subset=[target_col])
            if df.empty:
                raise ValueError(f"No valid labels left in dataset after dropping missing values in '{target_col}'")
            
        return df

    @staticmethod
    def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Cleans the dataset and splits it into feature matrix X and target vector y,
        excluding designated ID and operational target columns.
        """
        df_cleaned = DataService.clean_data(df, target_col)
        
        if target_col not in df_cleaned.columns:
            raise ValueError(f"Target column '{target_col}' not found in the dataset.")
            
        # Extract features and target
        y = df_cleaned[target_col]
        X = df_cleaned.drop(columns=[target_col])
        
        # Drop ID and target-overlapping columns specified in configuration
        cols_to_drop = [col for col in EXCLUDE_COLUMNS if col in X.columns]
        X = X.drop(columns=cols_to_drop)
        
        if X.empty:
            raise ValueError("No feature columns remaining after applying exclusion filters.")
            
        return X, y
