import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.inspection import permutation_importance

from backend.config import MODEL_PATH, DEFAULT_TARGET_COL

class ModelService:
    def __init__(self):
        self.model_artifact = None

    def load_model(self) -> bool:
        """
        Loads the trained model pipeline and metadata from the local joblib file.
        """
        if os.path.exists(MODEL_PATH):
            try:
                self.model_artifact = joblib.load(MODEL_PATH)
                print(f"Successfully loaded model trained at: {self.model_artifact.get('trained_at')}")
                return True
            except Exception as e:
                print(f"Error loading model from {MODEL_PATH}: {e}")
                return False
        return False

    def is_model_loaded(self) -> bool:
        return self.model_artifact is not None

    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns model metadata for configuration and dynamic UI rendering.
        """
        if not self.is_model_loaded():
            return {"status": "No model loaded"}
        
        return {
            "trained_at": self.model_artifact.get("trained_at"),
            "classes": self.model_artifact.get("class_names"),
            "features": self.model_artifact.get("feature_names"),
            "numeric_features": self.model_artifact.get("numeric_features"),
            "categorical_features": self.model_artifact.get("categorical_features"),
            "feature_defaults": self.model_artifact.get("feature_defaults"),
            "categorical_categories": self.model_artifact.get("categorical_categories"),
            "metrics": self.model_artifact.get("metrics", {}),
            "global_importances": self.model_artifact.get("global_importances", {})
        }

    def train_pipeline(self, df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COL) -> Dict[str, Any]:
        """
        Builds, preprocesses, trains, evaluates, and saves the classification pipeline.
        """
        from backend.services.data_service import DataService
        X, y = DataService.split_features_target(df, target_col)

        # Identify numeric and categorical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Compute defaults for runtime imputation (medians/modes)
        feature_defaults = {}
        for col in numeric_cols:
            feature_defaults[col] = float(X[col].median()) if not X[col].isnull().all() else 0.0
        for col in cat_cols:
            feature_defaults[col] = str(X[col].mode().iloc[0]) if not X[col].isnull().all() else ""

        # Preprocessing sub-pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Combined preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, cat_cols)
            ]
        )

        # Full training pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', HistGradientBoostingClassifier(
                class_weight='balanced', 
                random_state=42
            ))
        ])

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pipeline.fit(X_train, y_train)

        # Evaluate model
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Multiclass confusion matrix
        classes = pipeline.classes_.tolist()
        cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_).tolist()

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": {
                "classes": classes,
                "values": cm
            },
            "sample_size": len(df)
        }

        # Calculate model-agnostic global permutation importances on test set
        perm_res = permutation_importance(pipeline, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
        global_importances = {}
        for col, val in zip(X.columns, perm_res.importances_mean):
            global_importances[col] = float(max(0.0, val))
        global_importances = dict(sorted(global_importances.items(), key=lambda item: item[1], reverse=True))

        # Pack and persist model artifact
        self.model_artifact = {
            "pipeline": pipeline,
            "feature_names": list(X.columns),
            "numeric_features": numeric_cols,
            "categorical_features": cat_cols,
            "class_names": classes,
            "feature_defaults": feature_defaults,
            "categorical_categories": {col: X[col].dropna().unique().tolist() for col in cat_cols},
            "metrics": metrics,
            "global_importances": global_importances,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save to file
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(self.model_artifact, MODEL_PATH)
        print(f"Model saved successfully to {MODEL_PATH}")

        return self.get_metadata()

    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts defect risk and calculates local feature explanations for a single record.
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model is not loaded. Train a model first.")

        pipeline = self.model_artifact["pipeline"]
        feature_names = self.model_artifact["feature_names"]
        defaults = self.model_artifact["feature_defaults"]

        # Ensure all trained features are present, fill with training medians/modes if missing
        filled_record = {}
        for col in feature_names:
            filled_record[col] = record.get(col, defaults[col])

        # Convert to single-row DataFrame
        df_row = pd.DataFrame([filled_record])

        # Extract classification outputs
        prediction = str(pipeline.predict(df_row)[0])
        probabilities = pipeline.predict_proba(df_row)[0]
        class_names = self.model_artifact["class_names"]

        prob_dict = {str(c): float(p) for c, p in zip(class_names, probabilities)}
        confidence = prob_dict[prediction]

        # Calculate local explainability (SHAP / Fallback)
        explanation = self._explain_record(df_row, prediction, class_names.index(prediction))

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": prob_dict,
            "explanation": explanation
        }

    def _explain_record(self, df_row: pd.DataFrame, predicted_class: str, predicted_class_idx: int) -> List[Dict[str, Any]]:
        """
        Calculates local predictions drivers using SHAP TreeExplainer,
        and aggregates One-Hot encoded categoricals back to their raw features.
        """
        pipeline = self.model_artifact["pipeline"]
        preprocessor = pipeline.named_steps['preprocessor']
        classifier = pipeline.named_steps['classifier']
        
        categorical_features = self.model_artifact["categorical_features"]
        feature_names_out = preprocessor.get_feature_names_out()

        # Transform raw features into preprocessed structure for model
        x_preprocessed = preprocessor.transform(df_row)

        shap_explanation = []
        shap_calculated = False

        # Attempt SHAP calculation
        try:
            import shap
            # TreeExplainer works directly on the preprocessed representation that the Classifier receives
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(x_preprocessed)

            # In shap, for multiclass, it returns a list of shape arrays [n_samples, n_features] for each class
            # or an array of shape [n_samples, n_features, n_classes]
            if isinstance(shap_values, list):
                # Class specific SHAP contributions for the single row
                class_shap = shap_values[predicted_class_idx][0]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                class_shap = shap_values[0, :, predicted_class_idx]
            else:
                class_shap = shap_values[0] # Fallback for simple outputs

            # Aggregate preprocessed One-Hot categoricals back to their parent raw features
            raw_shap_sums = {}
            for feat_name, val in zip(feature_names_out, class_shap):
                raw_feat = feat_name
                if feat_name.startswith("num__"):
                    raw_feat = feat_name[len("num__"):]
                elif feat_name.startswith("cat__"):
                    remainder = feat_name[len("cat__"):]
                    raw_feat = remainder
                    # Find matching categorical parent name
                    for cat_col in categorical_features:
                        if remainder.startswith(cat_col + "_"):
                            raw_feat = cat_col
                            break
                
                raw_shap_sums[raw_feat] = raw_shap_sums.get(raw_feat, 0.0) + val

            # Sort features by absolute contribution
            sorted_raw_shap = sorted(raw_shap_sums.items(), key=lambda item: abs(item[1]), reverse=True)

            for feat, val in sorted_raw_shap[:5]:  # Top 5 factors
                effect = "increases predicted class" if val >= 0 else "decreases predicted class"
                shap_explanation.append({
                    "feature": feat,
                    "method": "SHAP TreeExplainer",
                    "contribution": float(round(val, 4)),
                    "effect": effect
                })
            shap_calculated = True
        except Exception as shap_error:
            print(f"SHAP local explainability calculation bypassed: {shap_error}")

        # Fallback to Global Feature Importance if SHAP fails or is missing
        if not shap_calculated:
            try:
                raw_importances = self.model_artifact.get("global_importances", {})
                
                # Sort global importances
                sorted_global = sorted(raw_importances.items(), key=lambda item: item[1], reverse=True)
                
                for feat, val in sorted_global[:5]:
                    shap_explanation.append({
                        "feature": feat,
                        "method": "Global Feature Importance (Fallback)",
                        "contribution": float(round(val, 4)),
                        "effect": "influences classification probability"
                    })
            except Exception as fallback_error:
                print(f"Fallback local explainability calculation failed: {fallback_error}")
                # Append a generic message so the UI can still display a safe state
                shap_explanation.append({
                    "feature": "N/A",
                    "method": "None Available",
                    "contribution": 0.0,
                    "effect": "No explanations could be generated"
                })

        return shap_explanation
