import sys

packages = [
    "fastapi",
    "uvicorn",
    "streamlit",
    "sklearn",
    "joblib",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "shap"
]

print("=== Checking Python Packages ===")
for pkg in packages:
    try:
        if pkg == "sklearn":
            import sklearn
        elif pkg == "shap":
            import shap
        else:
            __import__(pkg)
        print(f"  {pkg}: Installed")
    except ImportError:
        print(f"  {pkg}: NOT Installed")
