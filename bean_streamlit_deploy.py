from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Load the trained model and scaler
model = joblib.load('best_tuned_model.pkl')
scaler = joblib.load('transformation_artifacts.pkl')

FEATURE_BOUNDS = {
    "Area": (20420.0, 254616.0),
    "Perimeter": (524.0, 1986.0),
    "MajorAxisLength": (183.0, 739.0),
    "MinorAxisLength": (122.0, 461.0),
    "AspectRation": (1.0, 2.5),
    "Eccentricity": (0.2, 0.92),
    "ConvexArea": (20684.0, 263262.0),
    "EquivDiameter": (161.0, 570.0),
    "Extent": (0.55, 0.87),
    "Solidity": (0.91, 1.0),
    "roundness": (0.48, 1.0),
    "Compactness": (0.63, 1.0),
    "ShapeFactor1": (0.002, 0.011),
    "ShapeFactor2": (0.0005, 0.004),
    "ShapeFactor3": (0.40, 1.0),
    "ShapeFactor4": (0.94, 1.0),
}

st.set_page_config(page_title="Bean Classifier", page_icon="🫘")
st.title("Bean Classifier")
st.write("Enter the bean's physical measurements to predict its class.")

with st.form("bean_measurements"):
    values = {}
    columns = st.columns(2)

    for index, (feature, (lower, upper)) in enumerate(FEATURE_BOUNDS.items()):
        with columns[index % 2]:
            values[feature] = st.number_input(
                feature,
                min_value=lower,
                max_value=upper,
                value=(lower + upper) / 2,
                format="%.6f",
            )

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        if not MODEL_PATH.exists():
            st.error(f"Model file not found: {MODEL_PATH}")
        elif not TRANSFORMER_PATH.exists():
            st.error(f"Transformer file not found: {TRANSFORMER_PATH}")
        else:
            model = joblib.load(MODEL_PATH)
            transformer = joblib.load(TRANSFORMER_PATH)

            measurements = pd.DataFrame([values], columns=FEATURE_BOUNDS.keys())
            transformed = transformer.transform(measurements)

            prediction = model.predict(transformed)[0]
            st.success(f"Predicted class: **{prediction}**")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(transformed)[0]
                classes = getattr(model, "classes_", range(len(probabilities)))
                results = pd.DataFrame(
                    {"Class": classes, "Probability": probabilities}
                ).sort_values("Probability", ascending=False)

                st.subheader("Class probabilities")
                st.dataframe(results, hide_index=True)

    except Exception as error:
        st.error(f"Could not load the model or make a prediction: {error}")