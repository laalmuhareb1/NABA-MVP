import os
import requests
import streamlit as st

# ---- API base (FastAPI must be running on this URL/port) ----
API = os.getenv("NABA_API", "http://127.0.0.1:8010")

# ---- Fallback thresholds (used if the API doesn’t return per-condition "thr") ----
CLIENT_THRESHOLDS = {
    "hypertension": 0.50,
    "diabetes": 0.50,
    "dyslipidemia": 0.60,
}

st.set_page_config(page_title="NABA MVP — Risk & MES", layout="centered")
st.title("NABA MVP — Risk & MES")

with st.form("inp"):
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 0, 120, 35)
    bmi = c2.number_input("BMI", 0.0, 80.0, 27.3)
    systolic = c1.number_input("Systolic BP", 70, 220, 122)
    diastolic = c2.number_input("Diastolic BP", 40, 140, 80)

    hdl = c1.number_input("HDL (mg/dL)", 0.0, 150.0, 45.0)
    tg  = c2.number_input("Triglycerides (mg/dL)", 0.0, 1000.0, 160.0)
    a1c = c1.number_input("A1C (%)", 3.0, 15.0, 5.6)
    fpg = c2.number_input("Fasting Glucose (mg/dL)", 40.0, 400.0, 98.0)

    steps_y = c1.number_input("Steps / year", 0, 20_000_000, 2_000_000)
    cal_y   = c2.number_input("Calories / year", 0, 5_000_000, 800_000)

    on_statin = c1.checkbox("On statin", False)
    on_met    = c2.checkbox("On metformin", False)
    on_antiH  = c1.checkbox("On antihypertensive", False)

    submit = st.form_submit_button("Predict & Score")

if submit:
    # ---- Build payload
    features = {
        "age": age, "bmi": bmi, "systolic": systolic, "diastolic": diastolic,
        "hdl": hdl, "tg": tg, "a1c": a1c, "fasting_glucose": fpg,
        "steps_year_total": steps_y, "calories_year_total": cal_y,
        "on_statin": int(on_statin), "on_metformin": int(on_met),
        "on_antihypertensive": int(on_antiH),
    }

# ---- Call API safely
try:
    # Extended timeouts to prevent timeout errors on slow APIs
    pred = requests.post(
        f"{API}/predict",
        json={"features": features},
        timeout=60
    ).json()

    mes = requests.post(
        f"{API}/mes/score",
        json={"features": features},
        timeout=90
    ).json()

except requests.exceptions.Timeout:
    st.error(f"API at {API} took too long to respond (timeout). Please try again later.")
    st.stop()
except Exception as e:
    st.error(f"Could not reach API at {API}. Error: {e}")
    st.stop()


    # ---- Threshold legend
    st.markdown("### 🧪 Risk Thresholds (per condition)")
    st.markdown(
        """
        - **Hypertension:** ≥ **0.50**  
        - **Diabetes:** ≥ **0.50**  
        - **Dyslipidemia:** ≥ **0.60**
        """
    )
    st.divider()

    def active_thr(name: str) -> float:
        node = pred.get(name, {}) or {}
        return float(node.get("thr", CLIENT_THRESHOLDS.get(name, 0.5)))

    def pct(x) -> str:
        try:
            return f"{float(x):.2%}"
        except Exception:
            return "—"

    # ---- Risk Predictions
    st.subheader("Risk Predictions")
    cols = st.columns(3)

    htn_thr = active_thr("hypertension")
    dm_thr  = active_thr("diabetes")
    dlp_thr = active_thr("dyslipidemia")

    cols[0].metric(
        "Hypertension",
        pct(pred.get("hypertension", {}).get("prob", 0)),
        delta=f"Label: {pred.get('hypertension', {}).get('label', '—')} | Thr: {htn_thr:.2f}",
    )
    cols[1].metric(
        "Diabetes",
        pct(pred.get("diabetes", {}).get("prob", 0)),
        delta=f"Label: {pred.get('diabetes', {}).get('label', '—')} | Thr: {dm_thr:.2f}",
    )
    cols[2].metric(
        "Dyslipidemia",
        pct(pred.get("dyslipidemia", {}).get("prob", 0)),
        delta=f"Label: {pred.get('dyslipidemia', {}).get('label', '—')} | Thr: {dlp_thr:.2f}",
    )

    # ---- MES
    st.subheader("Metabolic Efficiency Score (MES)")
    st.metric("MES (0–100 ↑ better)", f"{mes.get('mes', 0):.1f}")
    with st.expander("Components & z-scores"):
        st.json({"parts": mes.get("parts", {}), "meta": mes.get("meta", {})})

    # ---- Top Drivers (most negative z-scores first)
    st.subheader("Top Drivers")
    parts = mes.get("parts", {}) or {}
    worst = sorted(parts.items(), key=lambda kv: kv[1])[:5]
    if not worst:
        st.info("No driver signals available.")
    else:
        for name, val in worst:
            st.write(f"- **{name}**: {val:+.2f}")

    # ---- Personalized Recommendations (via /advice), if available
    st.subheader("Personalized Recommendations")
    try:
        advice = requests.post(f"{API}/advice", json={"features": features}, timeout=15).json()
        tabs = st.tabs(["Patient", "Practitioner", "Organization", "Policy"])
        mapping = {0: "patient", 1: "practitioner", 2: "organization", 3: "policy"}
        for i, tab in enumerate(tabs):
            with tab:
                who = mapping[i]
                items = (advice.get("advice", {}) or {}).get(who, [])
                if not items:
                    st.info("No recommendations.")
                else:
                    for b in items:
                        st.markdown(f"- {b}")
    except Exception as e:
        st.warning(f"Advice service unavailable: {e}")

    # ---- Notes from API
    if isinstance(pred, dict) and pred.get("note"):
        st.caption(pred["note"])
