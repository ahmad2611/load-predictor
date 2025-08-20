#!/usr/bin/env python3
"""
Super-simple HVAC CLI

- Uses pre-trained models saved as joblib pickles (from your project).
- Choose which load to compute: HL (heating) or CL (cooling).
- Pass feature values on the command line.

Chosen project config (from the team's notebook, lowest error):
  * Model: PolynomialFeatures(degree=2) + LinearRegression
  * Features used for BOTH HL & CL: RC, SA, WA, RA, OH, GA

Examples:
  python hvac_simple_cli.py --mode HL --rc 2 --sa 771 --wa 679 --ra 368 --oh 19 --ga 3
  python hvac_simple_cli.py --mode CL --rc 2 --sa 771 --wa 679 --ra 368 --oh 19 --ga 3

By default, it looks for 'heating_model.pkl' and 'cooling_model.pkl' in the current directory.
Use --heat-model / --cool-model to point elsewhere.
"""
import argparse
import sys
import joblib
import numpy as np

FEATURES = ["RC", "SA", "WA", "RA", "OH", "GA"]

def main(argv=None):
    p = argparse.ArgumentParser(description="Super-simple HVAC Load CLI (uses pre-trained pickles)")
    p.add_argument("--mode", choices=["HL", "CL"], required=True, help="Which load to compute")
    p.add_argument("--rc", type=float, required=True, help="Relative compactness (RC)")
    p.add_argument("--sa", type=float, required=True, help="Surface area (SA)")
    p.add_argument("--wa", type=float, required=True, help="Wall area (WA)")
    p.add_argument("--ra", type=float, required=True, help="Roof area (RA)")
    p.add_argument("--oh", type=float, required=True, help="Overall height (OH)")
    p.add_argument("--ga", type=float, required=True, help="Glazing area (GA)")
    p.add_argument("--heat-model", default="heating_model.pkl", help="Path to heating model pickle")
    p.add_argument("--cool-model", default="cooling_model.pkl", help="Path to cooling model pickle")
    args = p.parse_args(argv)

    # 1 row in the expected order
    x = np.array([[args.rc, args.sa, args.wa, args.ra, args.oh, args.ga]], dtype=float)

    # Select the right model
    model_path = args.heat_model if args.mode == "HL" else args.cool_model
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        sys.stderr.write(
            f"Model file not found: {model_path}\n"
            f"Provide a valid --{'heat-model' if args.mode=='HL' else 'cool-model'} path, "
            f"or place your trained pickle in the current directory.\n"
        )
        sys.exit(2)

    # Predict and print a single scalar result
    yhat = model.predict(x)
    print(float(yhat[0]))

if __name__ == "__main__":
    main()
