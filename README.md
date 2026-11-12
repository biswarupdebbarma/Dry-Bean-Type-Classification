# Dry Bean Classification

A Streamlit app that predicts a dry bean variety from its physical measurements using a trained scikit-learn pipeline.

## Features

- Enter measurements manually and predict one bean variety.
- Upload a CSV to predict multiple rows at once.
- Download a CSV template or download the input rows with a `PredictedClass` column added.
- Review required CSV columns, their units, and input validation in the app.

## Setup

From the project directory, create and activate a virtual environment in PowerShell, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The trained model must be present at:

```text
saved_pickle_models/streamlit_deployment/best_tuned_model.pkl
```

## Run the app

```powershell
streamlit run bean_streamlit_deploy.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) to open in a browser.

## Predict from a CSV

1. Use **Download CSV template** in the app to get the required headers.
2. Provide one bean per row. Keep the column names exactly as shown below; feature columns must contain finite numeric values within their accepted ranges.
3. Upload the CSV and click **Predict uploaded CSV**.
4. Review the predictions and click **Download predictions as CSV**. The output preserves input columns and adds `PredictedClass`.

Additional input columns are preserved in the downloaded output but are not used by the model. The app reports missing columns, invalid numeric values, and out-of-range values before predicting.

| CSV column | Unit | Accepted range |
|---|---|---:|
| `Area` | px² | 20420–254616 |
| `Perimeter` | px | 524–1986 |
| `MajorAxisLength` | px | 183–739 |
| `MinorAxisLength` | px | 122–461 |
| `AspectRation` | unitless ratio | 1–2.5 |
| `Eccentricity` | unitless | 0.2–0.92 |
| `ConvexArea` | px² | 20684–263262 |
| `EquivDiameter` | px | 161–570 |
| `Extent` | unitless ratio | 0.55–0.87 |
| `Solidity` | unitless ratio | 0.91–1 |
| `roundness` | unitless ratio | 0.48–1 |
| `Compactness` | unitless ratio | 0.63–1 |
| `ShapeFactor1` | unitless | 0.002–0.011 |
| `ShapeFactor2` | unitless | 0.0005–0.004 |
| `ShapeFactor3` | unitless | 0.40–1 |
| `ShapeFactor4` | unitless | 0.94–1 |

`px` refers to pixels in the source image. Ratios and shape factors are dimensionless.
