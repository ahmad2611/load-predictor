
#!/usr/bin/env python3
"""
Train & Export HVAC Models

- Loads ENB2012data.csv (or a compatible CSV) and renames columns to the notebook's schema.
- Trains *separate* models for HL and CL using Polynomial(degree=2)+LinearRegression.
- For **HL**, evaluates two feature sets from the team's notebook and picks the one with lowest test RMSE:
    A) ['RC','SA','GA']
    B) ['RC','SA','WA','RA','OH','GA']
  For **CL**, uses ['RC','SA','WA','RA','OH','GA'] (the set used in the notebook).
- Saves: heating_model.pkl, cooling_model.pkl, hvac_meta.json (with RMSE and chosen features).

Usage:
  python train_export_hvac.py /path/to/ENB2012data.csv --out .
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

DEGREE = 2  # per notebook's best results

# Candidate feature sets observed in the notebook
HL_FEATURE_CANDIDATES = [
    ["RC", "SA", "GA"],
    ["RC", "SA", "WA", "RA", "OH", "GA"],
]
CL_FEATURES = ["RC", "SA", "WA", "RA", "OH", "GA"]

COL_MAP = {
    "X1": "RC", "X2": "SA", "X3": "WA", "X4": "RA",
    "X5": "OH", "X6": "O",  "X7": "GA", "X8": "GAD",
    "Y1": "HL", "Y2": "CL",
}

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})

def build_pipeline():
    return Pipeline([
        ("poly", PolynomialFeatures(degree=DEGREE, include_bias=True)),
        ("lin", LinearRegression())
    ])

def eval_feature_set(df: pd.DataFrame, features: list, target: str, random_state: int = 42):
    X = df[features].copy()
    y = df[target].astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=random_state)
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    yhat_tr = pipe.predict(X_tr)
    yhat_te = pipe.predict(X_te)
    rmse_tr = float(np.sqrt(mean_squared_error(y_tr, yhat_tr)))
    rmse_te = float(np.sqrt(mean_squared_error(y_te, yhat_te)))
    return {"model": pipe, "features": features, "rmse_train": rmse_tr, "rmse_test": rmse_te}

def main():
    ap = argparse.ArgumentParser(description="Train & export HVAC HL/CL models (degree=2)")
    ap.add_argument("csv", help="Path to ENB2012data.csv or compatible CSV")
    ap.add_argument("--out", default=".", help="Output directory (default: .)")
    ap.add_argument("--random-state", type=int, default=42, help="Reproducible split (default: 42)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    df = normalize_headers(df)
    required_targets = ["HL", "CL"]
    for t in required_targets:
        if t not in df.columns:
            raise ValueError(f"CSV must include target column '{t}'. Found columns: {list(df.columns)}")

    # HEATING: pick the best of the two candidate feature sets
    hl_results = []
    for feats in HL_FEATURE_CANDIDATES:
        missing = [c for c in feats if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for HL features {feats}: {missing}")
        res = eval_feature_set(df, feats, target="HL", random_state=args.random_state)
        hl_results.append(res)
    best_hl = min(hl_results, key=lambda r: r["rmse_test"])  # choose by test RMSE

    # COOLING: use the notebook's feature set
    missing = [c for c in CL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for CL features {CL_FEATURES}: {missing}")
    best_cl = eval_feature_set(df, CL_FEATURES, target="CL", random_state=args.random_state)

    # Persist the pipelines
    heat_path = os.path.join(args.out, "heating_model.pkl")
    cool_path = os.path.join(args.out, "cooling_model.pkl")
    joblib.dump(best_hl["model"], heat_path)
    joblib.dump(best_cl["model"], cool_path)

    meta = {
        "degree": DEGREE,
        "heating": {
            "features": best_hl["features"],
            "rmse_train": best_hl["rmse_train"],
            "rmse_test": best_hl["rmse_test"],
            "model_path": heat_path,
        },
        "cooling": {
            "features": best_cl["features"],
            "rmse_train": best_cl["rmse_train"],
            "rmse_test": best_cl["rmse_test"],
            "model_path": cool_path,
        },
    }
    meta_path = os.path.join(args.out, "hvac_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("== Export complete ==")
    print(f"Heating  -> {heat_path}") 
    print(f"Cooling  -> {cool_path}")
    print(f"Metadata -> {meta_path}")
    print("\nSummary:")
    print(f"  HL: features={best_hl['features']}  RMSE(train)={best_hl['rmse_train']:.3f}  RMSE(test)={best_hl['rmse_test']:.3f}")
    print(f"  CL: features={best_cl['features']}  RMSE(train)={best_cl['rmse_train']:.3f}  RMSE(test)={best_cl['rmse_test']:.3f}")

if __name__ == "__main__":
    main()
