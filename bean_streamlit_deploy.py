import io

import joblib
import pandas as pd
import streamlit as st

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

model_file = best_tuned_model.pkl,
    ,
)
transformer_file =transformation_artifacts.pkl,
    ,
)

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
    if model_file is None:
        st.error("Upload best_tuned_model.pkl to make a prediction.")
    else:
        try:
            model = joblib.load(io.BytesIO(model_file.getvalue()))
            measurements = pd.DataFrame([values])

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