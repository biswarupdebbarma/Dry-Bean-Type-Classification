from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_tuned_model.pkl"

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

FEATURE_UNITS = {
    "Area": "px²",
    "Perimeter": "px",
    "MajorAxisLength": "px",
    "MinorAxisLength": "px",
    "AspectRation": "unitless ratio",
    "Eccentricity": "unitless",
    "ConvexArea": "px²",
    "EquivDiameter": "px",
    "Extent": "unitless ratio",
    "Solidity": "unitless ratio",
    "roundness": "unitless ratio",
    "Compactness": "unitless ratio",
    "ShapeFactor1": "unitless",
    "ShapeFactor2": "unitless",
    "ShapeFactor3": "unitless",
    "ShapeFactor4": "unitless",
}


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def validate_csv_measurements(dataframe):
    missing_columns = [
        feature for feature in FEATURE_BOUNDS if feature not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )
    if dataframe.empty:
        raise ValueError("The CSV must contain at least one data row.")

    measurements = dataframe.loc[:, list(FEATURE_BOUNDS)].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid_values = measurements.isna() | ~np.isfinite(measurements)
    if invalid_values.any().any():
        invalid_cells = [
            f"row {row_index + 2}, {column}"
            for row_index, row in invalid_values.iterrows()
            for column, is_invalid in row.items()
            if is_invalid
        ]
        raise ValueError(
            "Values must be numeric and finite. Check "
            + "; ".join(invalid_cells[:10])
        )

    out_of_range = []
    for feature, (lower, upper) in FEATURE_BOUNDS.items():
        invalid_rows = measurements.index[
            (measurements[feature] < lower) | (measurements[feature] > upper)
        ]
        out_of_range.extend(
            f"row {row_index + 2}, {feature} (allowed {lower}–{upper})"
            for row_index in invalid_rows
        )
    if out_of_range:
        raise ValueError("Values outside allowed ranges: " + "; ".join(out_of_range[:10]))

    return measurements


st.set_page_config(page_title="Bean Classifier", page_icon="🫘")

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 8% 8%, rgba(218, 164, 87, .13), transparent 30%),
            linear-gradient(160deg, #f8f6ef 0%, #f1f3eb 55%, #f8f4eb 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1080px; padding-top: 2.2rem; padding-bottom: 3rem; }
    .hero {
        position: relative;
        overflow: hidden;
        min-height: 230px;
        padding: 38px 42px;
        border: 1px solid rgba(255,255,255,.16);
        border-radius: 28px;
        color: #fffaf0;
        background: linear-gradient(115deg, #173b32 0%, #245a46 58%, #39734f 100%);
        box-shadow: 0 18px 45px rgba(28, 62, 47, .18);
    }
    .hero-copy { position: relative; z-index: 2; max-width: 620px; }
    .eyebrow {
        margin: 0 0 12px;
        color: #e7c98e;
        font-size: .76rem;
        font-weight: 750;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: 0;
        color: #fffaf0;
        font-size: clamp(2.2rem, 5vw, 3.5rem);
        font-weight: 760;
        letter-spacing: -.045em;
        line-height: 1.05;
    }
    .hero p {
        max-width: 490px;
        margin: 14px 0 0;
        color: #dce8d9;
        font-size: 1.04rem;
        line-height: 1.6;
    }
    .bean-art {
        position: absolute;
        top: 50%;
        right: 9%;
        width: 180px;
        height: 180px;
        transform: translateY(-50%);
        border: 1px solid rgba(255,255,255,.13);
        border-radius: 50%;
        background: rgba(255,255,255,.045);
    }
    .bean {
        position: absolute;
        width: 72px;
        height: 104px;
        border: 1px solid rgba(255,255,255,.22);
        border-radius: 54% 46% 52% 48% / 43% 46% 54% 57%;
        box-shadow: inset 8px 4px 14px rgba(255,255,255,.16), 0 12px 24px rgba(0,0,0,.2);
    }
    .bean::after {
        position: absolute;
        top: 14%;
        left: 47%;
        width: 2px;
        height: 72%;
        border-radius: 50%;
        background: rgba(255,255,255,.6);
        content: "";
        transform: rotate(12deg);
    }
    .bean-one { top: 20px; left: 26px; background: linear-gradient(135deg, #c7864d, #8b4e37); transform: rotate(-32deg); }
    .bean-two { top: 46px; right: 22px; background: linear-gradient(135deg, #e1b46c, #a56d3f); transform: rotate(31deg); }
    .bean-three { right: 54px; bottom: -16px; width: 60px; height: 86px; background: linear-gradient(135deg, #ac6745, #743f35); transform: rotate(8deg); }
    .stats { margin: 20px 0 30px; }
    .stat-card {
        min-height: 94px;
        padding: 17px 20px;
        border: 1px solid #e6e5d9;
        border-radius: 18px;
        background: rgba(255,255,255,.78);
        box-shadow: 0 7px 20px rgba(47, 60, 42, .045);
    }
    .stat-value { color: #28523e; font-size: 1.55rem; font-weight: 760; line-height: 1.2; }
    .stat-label { margin-top: 6px; color: #697568; font-size: .85rem; }
    .section-heading { margin: 0 0 4px; color: #203d31; font-size: 1.45rem; font-weight: 720; }
    .section-note { margin: 0 0 16px; color: #687467; }
    [data-testid="stNumberInput"] label { color: #34483b; font-weight: 620; }
    [data-testid="stNumberInput"] input {
        border-radius: 11px;
        border-color: #dce2d7;
        background: #fff;
    }
    [data-testid="stFormSubmitButton"] button {
        min-height: 3.1rem;
        border: 0;
        border-radius: 12px;
        background: linear-gradient(100deg, #285b42, #38754e);
        color: white;
        font-size: 1rem;
        font-weight: 700;
        box-shadow: 0 8px 18px rgba(40, 91, 66, .18);
    }
    [data-testid="stFormSubmitButton"] button:hover { border: 0; background: #214b38; color: white; }
    .page-footer { margin-top: 28px; color: #788073; font-size: .82rem; text-align: center; }
    @media (max-width: 640px) {
        .hero { min-height: 220px; padding: 30px 24px; }
        .bean-art { right: -70px; opacity: .43; }
        .hero-copy { max-width: 100%; }
    }
    </style>
    <section class="hero" aria-label="Dry bean classifier">
      <div class="hero-copy">
        <div class="eyebrow">From measurements to insight</div>
        <h1>Meet your<br>bean classifier.</h1>
        <p>Discover the likely dry bean variety from its physical characteristics, powered by a trained machine-learning model.</p>
      </div>
      <div class="bean-art" aria-hidden="true">
        <span class="bean bean-one"></span>
        <span class="bean bean-two"></span>
        <span class="bean bean-three"></span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="stats">', unsafe_allow_html=True)
stat_columns = st.columns(3)
for column, value, label in zip(
    stat_columns,
    ("16", "7", "1"),
    ("Physical measurements", "Dry bean varieties", "Instant prediction"),
):
    with column:
        st.markdown(
            f'<div class="stat-card"><div class="stat-value">{value}</div>'
            f'<div class="stat-label">{label}</div></div>',
            unsafe_allow_html=True,
        )
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-heading">Bean measurements</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-note">Enter the physical measurements below to identify the closest variety.</div>',
    unsafe_allow_html=True,
)

with st.form("bean_measurements"):
    values = {}
    columns = st.columns(2)

    for index, (feature, (lower, upper)) in enumerate(FEATURE_BOUNDS.items()):
        with columns[index % 2]:
            values[feature] = st.number_input(
                f"{feature} ({FEATURE_UNITS[feature]})",
                min_value=lower,
                max_value=upper,
                value=(lower + upper) / 2,
                format="%.6f",
            )

    submitted = st.form_submit_button(
        "Predict bean variety", type="primary", use_container_width=True
    )

if submitted:
    try:
        model = load_model()
        measurements = pd.DataFrame([values], columns=FEATURE_BOUNDS.keys())
        prediction = model.predict(measurements)[0]
        st.success(f"Predicted class: **{prediction}**")

    except Exception as error:
        st.error(f"Could not load the model or make a prediction: {error}")

st.markdown('<div class="section-heading">Predict a CSV file</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-note">Upload a CSV with one bean per row and all 16 measurement columns. '
    'Your downloaded file will include a predicted class for each row.</div>',
    unsafe_allow_html=True,
)
with st.expander("View required columns and units"):
    st.dataframe(
        pd.DataFrame(
            {
                "CSV column": list(FEATURE_UNITS),
                "Measurement unit": list(FEATURE_UNITS.values()),
            }
        ),
        hide_index=True,
        width="stretch",
    )

template = pd.DataFrame(
    [{feature: (lower + upper) / 2 for feature, (lower, upper) in FEATURE_BOUNDS.items()}]
)
st.download_button(
    "Download CSV template",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name="bean_measurements_template.csv",
    mime="text/csv",
)

uploaded_csv = st.file_uploader(
    "Upload measurements CSV", type=["csv"], key="bean_measurements_csv"
)
if st.button("Predict uploaded CSV", type="primary", key="predict_csv"):
    if uploaded_csv is None:
        st.error("Choose a CSV file before predicting.")
    else:
        try:
            input_dataframe = pd.read_csv(uploaded_csv)
            csv_measurements = validate_csv_measurements(input_dataframe)
            batch_predictions = load_model().predict(csv_measurements)
            output_dataframe = input_dataframe.copy()
            output_dataframe["PredictedClass"] = batch_predictions

            st.success(f"Predicted {len(output_dataframe)} bean rows.")
            st.dataframe(output_dataframe, width="stretch")
            st.download_button(
                "Download predictions as CSV",
                data=output_dataframe.to_csv(index=False).encode("utf-8-sig"),
                file_name="bean_predictions.csv",
                mime="text/csv",
                key="download_predictions",
            )
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
            st.error(f"Could not process the uploaded CSV: {error}")
        except Exception as error:
            st.error(f"Could not load the model or predict the CSV: {error}")

st.markdown(
    '<div class="page-footer">A little science behind every bean 🫘</div>',
    unsafe_allow_html=True,
)