# ==========================================================
# CKD Risk Prediction Dashboard
# ==========================================================

import os
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.inference import (
    load_bundle,
    predict_sample,
    preprocess_input,
)

from src.evaluation import (
    explain_model,
    explain_single_prediction,
)

# ==========================================================
# PAGE CONFIG
# ==========================================================

_tab_icon = "assets/logo.png" if os.path.exists("assets/logo.png") else "🩺"

st.set_page_config(
    page_title="CKD Risk Prediction Dashboard",
    page_icon=_tab_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# LOAD CSS
# ==========================================================

css_path = "assets/style.css"

if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )

# ==========================================================
# LOAD MODEL (single copy)
# ==========================================================

@st.cache_resource
def get_bundle():
    return load_bundle("models/ckd_pipeline.joblib")


bundle = get_bundle()

# ==========================================================
# GAUGE HELPER
# ==========================================================

def render_gauge(ckd_probability: float, prediction_label: str):
    """
    Renders a circular Plotly gauge showing the ACTUAL CKD probability
    (0-100%), always in the same direction: higher % = higher CKD risk.
    """
    pct = ckd_probability * 100

    bar_color = "#e74c3c" if prediction_label == "ckd" else "#2ecc71"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 40}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "white",
                "borderwidth": 1,
                "steps": [
                    {"range": [0, 30], "color": "#d4f8e0"},
                    {"range": [30, 70], "color": "#fff3cd"},
                    {"range": [70, 100], "color": "#fddede"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": pct,
                },
            },
        )
    )

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=30, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_confidence_badge(display_value: float, prediction: str):
    """
    Shows a single confidence badge based on display_value (confidence in
    the PREDICTED class).
    """
    if display_value >= 0.90:
        label = "Very High Confidence"
        if prediction == "ckd":
            st.error(label)
        else:
            st.success(label)

    elif display_value >= 0.75:
        label = "High Confidence"
        if prediction == "ckd":
            st.warning(label)
        else:
            st.info(label)

    else:
        label = "Moderate Confidence"
        if prediction == "ckd":
            st.info(label)
        else:
            st.warning(label)


# ==========================================================
# SIDEBAR (single copy)
# ==========================================================

with st.sidebar:

    logo_path = "assets/logo.png"

    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)

    st.markdown("## CKD Prediction")
    st.caption("AI Powered Clinical Decision Support")

    st.divider()

    st.subheader("🧠 Model")

    st.metric("Algorithm", bundle["model_name"].upper())

    st.markdown("##### MLflow Run")
    st.code(bundle["mlflow_run_id"][:8], language=None)

    st.markdown("##### Prediction Type")
    st.info("Binary Classification")

    st.divider()

    st.subheader("📖 About")
    st.info(
        """
This application predicts the likelihood of **Chronic Kidney Disease (CKD)** using laboratory values and clinical findings.

It is intended for educational and clinical decision-support purposes only.
"""
    )

    st.divider()

    st.subheader("✅ Features")
    st.success(
        """
✔ XGBoost

✔ SHAP Explainability

✔ MLflow Tracking

✔ Streamlit Dashboard

✔ Clinical Decision Support
"""
    )

# ==========================================================
# HEADER (single copy)
# ==========================================================

header_col1, header_col2 = st.columns([0.5, 5], gap="small", vertical_alignment="center")

with header_col1:
    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=180)

with header_col2:
    st.markdown(
        """
# CKD Risk Prediction Dashboard

### AI Powered Clinical Decision Support System
"""
    )
    st.markdown(
        """
Predict the likelihood of **Chronic Kidney Disease (CKD)** using
laboratory values and clinical findings with an explainable
machine learning model.
"""
    )

# ==========================================================
# DISCLAIMER
# ==========================================================

st.warning(
    """
### ⚠ Medical Disclaimer

This application is intended for **clinical decision support only**.

It estimates CKD risk using a trained machine learning model.

**It is NOT a medical diagnosis and should never replace a qualified healthcare professional.**
"""
)

# ==========================================================
# DASHBOARD OVERVIEW
# ==========================================================

st.markdown("## 📊 Dashboard Overview")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Model", bundle["model_name"].upper())

with m2:
    st.metric("Input Features", "26")

with m3:
    st.metric("Explainability", "SHAP")

with m4:
    st.metric("Status", "🟢 Ready")

st.divider()

# ==========================================================
# PATIENT INFORMATION
# ==========================================================

st.markdown(
    """
# 👨‍⚕️ Patient Clinical Information

Enter the patient's laboratory values and clinical findings below.
"""
)

# ==========================================================
# PATIENT INPUT FORM
# ==========================================================

with st.form("patient_form"):

    st.markdown("## 📋 Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        with st.expander("👤 Patient Information", expanded=True):
            age = st.number_input("Age (Years)", min_value=0, max_value=120, value=45)
            bp = st.number_input("Blood Pressure (mm/Hg)", min_value=40, max_value=250, value=80)

        with st.expander("🧪 Urinalysis", expanded=True):
            sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=3)
            al = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5])
            su = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5])
            bgr = st.number_input("Random Blood Glucose (mg/dL)", min_value=20, max_value=500, value=120)

    with col2:

        with st.expander("🩸 Blood Chemistry", expanded=True):
            bu = st.number_input("Blood Urea (mg/dL)", min_value=1.0, max_value=400.0, value=30.0)
            sc = st.number_input("Serum Creatinine (mg/dL)", min_value=0.1, max_value=30.0, value=1.0)
            sod = st.number_input("Sodium (mEq/L)", min_value=100.0, max_value=170.0, value=140.0)
            pot = st.number_input("Potassium (mEq/L)", min_value=1.0, max_value=30.0, value=4.5)

        with st.expander("🧬 Blood Count", expanded=True):
            hemo = st.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=20.0, value=13.5)
            pcv = st.number_input("Packed Cell Volume (%)", min_value=10.0, max_value=60.0, value=40.0)
            wc = st.number_input("White Blood Cell Count", min_value=2000.0, max_value=25000.0, value=8000.0)
            rc = st.number_input("Red Blood Cell Count", min_value=2.0, max_value=8.0, value=5.0)

    with col3:

        with st.expander("🩺 Clinical Findings", expanded=True):
            rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
            pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
            pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])
            ba = st.selectbox("Bacteria", ["notpresent", "present"])
            htn = st.selectbox("Hypertension", ["no", "yes"])
            dm = st.selectbox("Diabetes Mellitus", ["no", "yes"])
            cad = st.selectbox("Coronary Artery Disease", ["no", "yes"])
            appet = st.selectbox("Appetite", ["good", "poor"])
            pe = st.selectbox("Pedal Edema", ["no", "yes"])
            ane = st.selectbox("Anemia", ["no", "yes"])

    st.divider()
    st.subheader("📊 Current Patient Summary")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Age", age)
    with s2:
        st.metric("Blood Pressure", bp)
    with s3:
        st.metric("Creatinine", sc)
    with s4:
        st.metric("Hemoglobin", hemo)

    submitted = st.form_submit_button(
        "🩺 Predict CKD Risk",
        use_container_width=True,
        type="primary",
    )

# ==========================================================
# PREDICTION
# ==========================================================

if submitted:

    raw_input = {
        "age": age, "bp": bp, "sg": sg, "al": al, "su": su,
        "rbc": rbc, "pc": pc, "pcc": pcc, "ba": ba,
        "bgr": bgr, "bu": bu, "sc": sc, "sod": sod, "pot": pot,
        "hemo": hemo, "pcv": pcv, "wc": wc, "rc": rc,
        "htn": htn, "dm": dm, "cad": cad, "appet": appet, "pe": pe, "ane": ane,
    }

    with st.spinner("🧠 AI is analysing the patient's data..."):
        progress = st.progress(0)
        for i in range(100):
            progress.progress(i + 1)
            time.sleep(0.01)

        result = predict_sample(raw_input, bundle=bundle)

    progress.empty()

    processed = preprocess_input(raw_input, bundle)
    explainer, shap_values = explain_model(bundle["model"], processed)
    explanation = explain_single_prediction(bundle["model"], explainer, shap_values, processed, 0)

    prediction = result["prediction_label"]
    probability = result["probability"]

    st.divider()
    st.markdown("# 📋 Prediction Result")

    result_col, explain_col = st.columns([1, 2])

    with result_col:

        st.subheader("🎯 Prediction")

        if prediction == "ckd":
            st.error("## 🔴 High Risk of CKD")
            confidence_value = probability
        else:
            st.success("## 🟢 Low Risk of CKD")
            confidence_value = 1 - probability

        render_gauge(probability, prediction)
        render_confidence_badge(confidence_value, prediction)

        st.divider()
        st.subheader("📋 Clinical Recommendation")

        if prediction == "ckd":
            st.warning(
                """
• Consult a nephrologist.

• Repeat kidney function tests.

• Monitor blood pressure regularly.

• Review diabetes status.

• Follow physician recommendations.
"""
            )
        else:
            st.success(
                """
• Continue regular health check-ups.

• Maintain a healthy lifestyle.

• Stay hydrated.

• Exercise regularly.

• Monitor kidney function annually.
"""
            )

    with explain_col:

        st.subheader("🧠 Why did the AI make this prediction?")
        st.caption("Top features contributing to the prediction.")

        contributions = explanation["top_contributions"][:5]

        max_abs_shap = max(abs(item["shap_contribution"]) for item in contributions) or 1.0

        for item in contributions:
            feature = item["feature"]
            shap_value = item["shap_contribution"]

            display_val = raw_input.get(feature, item["value"])

            if shap_value >= 0:
                icon = "🔴"
                direction = "Increases CKD Risk"
            else:
                icon = "🟢"
                direction = "Decreases CKD Risk"

            if isinstance(display_val, (int, float)):
                value_str = f"{display_val:.2f}"
            else:
                value_str = str(display_val)

            with st.container():
                st.markdown(
                    f"""
### {icon} {feature}

**Patient Value:** `{value_str}`

**Impact:** {direction}
"""
                )
                st.progress(min(abs(shap_value) / max_abs_shap, 1.0))

    st.divider()
    st.header("📋 Patient Summary")

    summary = pd.DataFrame(
        {
            "Feature": [
                "Age", "Blood Pressure", "Blood Glucose", "Blood Urea",
                "Serum Creatinine", "Hemoglobin", "Sodium", "Potassium",
            ],
            "Value": [age, bp, bgr, bu, sc, hemo, sod, pot],
        }
    )

    st.dataframe(summary, use_container_width=True, hide_index=True)

    report = pd.DataFrame(
        {
            "Prediction": ["CKD" if prediction == "ckd" else "Not CKD"],
            "Probability": [probability],
            "Age": [age],
            "Blood Pressure": [bp],
            "Blood Glucose": [bgr],
            "Blood Urea": [bu],
            "Serum Creatinine": [sc],
            "Hemoglobin": [hemo],
        }
    )

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction Report (CSV)",
        data=csv,
        file_name="ckd_prediction_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("ℹ️ Model Information"):
        st.markdown(f"**Model:** {bundle['model_name']}")
        st.markdown(f"**MLflow Run ID:** `{bundle['mlflow_run_id']}`")
        st.markdown(
            """
**Pipeline**

- Raw clinical inputs
- Feature Engineering
- Standard Scaling
- XGBoost Classification
- SHAP Explainability
"""
        )

    st.divider()
    st.caption(
        """
🩺 **CKD Risk Prediction Dashboard**

Built using **Python**, **Streamlit**, **XGBoost**, **SHAP**, and **MLflow**.

This tool is intended for educational and clinical decision-support purposes only and should not be used as a substitute for professional medical diagnosis.
"""
    )

# ==========================================================
# AI ASSISTANT (Day 16 -- real prediction-explainer wired in)
# ==========================================================

st.divider()
st.markdown("## 🤖 AI Health Assistant")
st.caption("Ask why the model made its last prediction, or general CKD reference questions.")

# Store the last prediction's context so follow-up questions can reference it
if submitted:
    st.session_state["last_prediction"] = result
    st.session_state["last_shap_explanation"] = explanation

user_question = st.chat_input(
    "Ask about this prediction or general CKD info...",
    key="ai_assistant_chat",
)

if user_question:
    from src.ai_assistant.router import route_message

    context = {
        "last_prediction": st.session_state.get("last_prediction"),
        "last_shap_explanation": st.session_state.get("last_shap_explanation"),
    }

    route_result = route_message(user_question, context=context)

    if route_result["tool"] == "guardrail_refusal":
        st.warning(route_result["message"])
    elif route_result["tool"] == "shap_explainer":
        st.chat_message("assistant").write(route_result["explanation"])
        if not route_result["grounded"]:
            st.error(
                f"⚠️ Grounding check failed -- this explanation may reference "
                f"numbers not in the source data: {route_result['ungrounded_numbers']}"
            )
    elif route_result["tool"] == "rag":
        st.info("General reference Q&A not yet implemented (Day 17).")
    else:
        st.info("I couldn't confidently match this to a specific capability yet.")