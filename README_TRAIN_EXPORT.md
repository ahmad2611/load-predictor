
# Train & Export (one-time)

This script trains the **final models** and exports them as pickles the CLI can load.

**What it does**
- Reads your ENB dataset and aligns column names to the notebook's schema.
- Model: `PolynomialFeatures(degree=2) + LinearRegression`.
- **Heating (HL):** tries two notebook feature sets and picks the one with the **lowest test RMSE**:
    - A) `['RC','SA','GA']`
    - B) `['RC','SA','WA','RA','OH','GA']`
- **Cooling (CL):** uses `['RC','SA','WA','RA','OH','GA']` (as in the notebook).
- Writes:
    - `heating_model.pkl`
    - `cooling_model.pkl`
    - `hvac_meta.json` (chosen features + RMSEs)

## Requirements
```bash
pip install scikit-learn pandas numpy joblib
```

## Run
```bash
python train_export_hvac.py /path/to/ENB2012data.csv --out .
```

When done, use the simple CLI to predict:
```bash
# Heating
python hvac_simple_cli.py --mode HL --rc 2 --sa 771 --wa 679 --ra 368 --oh 19 --ga 3

# Cooling
python hvac_simple_cli.py --mode CL --rc 2 --sa 771 --wa 679 --ra 368 --oh 19 --ga 3
```
