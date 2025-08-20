# HVAC Load Prediction Project

## What does this project do?
Predicts **heating load (HL)** and **cooling load (CL)** of homes using a trained regression model, and provides a simple command-line interface (CLI) for quick predictions.

## Why did I make it?
This was part of a project to practice:
- Building and evaluating regression models from real data (ENB2012 dataset).
- Selecting the model with the **lowest train/test error**.
- Packaging results into a simple, usable CLI tool.  

I wanted to deeply learn how to train and create and choose models. I also wanted to explore how to turn them into something anyone can run with a single command.

## How it works
1. **Model Construction (notebook)**  
   Explored polynomial regression models, compared errors, and selected the best configuration:
   - Algorithm: `PolynomialFeatures(degree=2) + LinearRegression`
   - HL features tested: `['RC','SA','GA']` vs `['RC','SA','WA','RA','OH','GA']` → chose the one with **lowest RMSE**
   - CL features: `['RC','SA','WA','RA','OH','GA']`

2. **Training & Export (script)**  
   Run once to train and export the chosen models:
   ```bash
   python train_export_hvac.py ENB2012data.csv --out .
   # produces heating_model.pkl, cooling_model.pkl, hvac_meta.json
   ```

3. **Prediction (CLI tool)**  
   Use the simple CLI to compute HL or CL from inputted values:
   ```bash
   # Heating
   python hvac_simple_cli.py --mode HL --rc 2 --sa 771 --wa 679 --ra 368 --oh 19 --ga 3

   # Cooling
   python hvac_simple_cli.py --mode CL --rc 2 --sa 771 --wa 679 --ra 368 --oh 19 --ga 3
   ```

   Output: a single numeric prediction.

## Setup
```bash
pip install scikit-learn pandas numpy joblib
```

## What I learned
- How to compare and select models based on **train/test RMSE**, avoiding overfitting.  
- How to move from **research code** (notebooks) to **usable software** (CLI).  
- How to simplify a workflow so others can run predictions without touching the model training process.  
