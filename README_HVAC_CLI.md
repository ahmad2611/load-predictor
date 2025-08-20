# HVAC Simple CLI

Minimal command-line tool to compute **Heating Load (HL)** or **Cooling Load (CL)** from your **pre-trained** models.

## Requirements
- Python 3.9+
- `scikit-learn` (needed to unpickle the pipeline)
- `joblib`
- `numpy`

```bash
pip install scikit-learn joblib numpy
```

## Files expected
Place your trained pickles in the same folder (or pass custom paths):
- `heating_model.pkl`
- `cooling_model.pkl`

> These should be pipelines for: PolynomialFeatures(degree=2) + LinearRegression, trained on features `RC, SA, WA, RA, OH, GA`.

## Usage

### Heating (HL)
```bash
python hvac_simple_cli.py --mode HL   --rc 2 --sa 771 --wa 679 --ra 368 --oh 19 --ga 3
```

### Cooling (CL)
```bash
python hvac_simple_cli.py --mode CL   --rc 2 --sa 771 --wa 679 --ra 368 --oh 19 --ga 3
```

### Custom model paths
```bash
python hvac_simple_cli.py --mode HL --heat-model ./heating_model.pkl --rc ... --sa ... --wa ... --ra ... --oh ... --ga ...
python hvac_simple_cli.py --mode CL --cool-model ./cooling_model.pkl --rc ... --sa ... --wa ... --ra ... --oh ... --ga ...
```

**Output:** a single numeric prediction printed to stdout.
