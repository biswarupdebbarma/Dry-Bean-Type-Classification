import io

import joblib
import pandas as pd
import streamlit as st

FEATURES = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]

st.set_page_config(page_title="Bean Classifier", page_icon="🫘")
st.title("Bean Classifier")
st.write("Enter the bean's physical measurements to predict its class.")

model_file = st.file_uploader(
    "Select best_tuned_model.pkl",
    type=["pkl", "pickle"],
)
transformer_file = st.file_uploader(
    "Optional: select a separately saved, fitted transformer",
    type=["pkl", "pickle"],
)

with st.form("bean_measurements"):
    values = {}
    columns = st.columns(2)

    for index, feature in enumerate(FEATURES):
        with columns[index % 2]:
            values[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f",
            )

    submitted = st.form_submit_button("Predict")

if submitted:
    if model_file is None:
        st.error("Upload best_tuned_model.pkl to make a prediction.")
    else:
        try:
            model = joblib.load(io.BytesIO(model_file.getvalue()))
            measurements = pd.DataFrame([values], columns=FEATURES)

            if transformer_file is not None:
                transformer = joblib.load(
                    io.BytesIO(transformer_file.getvalue())
                )
                measurements = transformer.transform(measurements)

            prediction = model.predict(measurements)[0]
            st.success(f"Predicted class: **{prediction}**")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(measurements)[0]
                classes = getattr(model, "classes_", range(len(probabilities)))
                results = pd.DataFrame(
                    {"Class": classes, "Probability": probabilities}
                ).sort_values("Probability", ascending=False)
                st.subheader("Class probabilities")
                st.dataframe(results, hide_index=True)

        except Exception as error:
            st.error(f"Could not load the model or make a prediction: {error}")