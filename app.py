# app.py — NABA MVP (EN/AR) — tabbed UI

import streamlit as st
import sqlite3
from datetime import datetime
import pandas as pd
from naba_core import Inputs, compute_mes, predict_dsi, recommendations, bmi

# ---------------------- Page & Theme ----------------------
st.set_page_config(page_title="NABA MVP", page_icon="🧬", layout="wide")

# ---------------------- Language pack ----------------------
LANG = {
    "en": {
        "title": "NABA — MVP (Metabolic Efficiency & Disease Susceptibility)",
        "caption": "Datathon demo: computes MES and nutrition-related disease susceptibility (HTN/T2D). Not for clinical use.",
        "lang_label": "Language",
        "lang_en": "English",
        "lang_ar": "العربية",
        "inputs_hdr": "Inputs",
        "age": "Age",
        "sex": "Sex",
        "sex_m": "M",
        "sex_f": "F",
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "steps": "Steps (per day)",
        "sleep": "Sleep (hours)",
        "cal_in": "Calories Intake (kcal)",
        "muscle": "Muscle %",
        "sbp": "Systolic BP (mmHg)",
        "glu": "Fasting Glucose (mg/dL)",
        "sodium": "Sodium Intake (mg/day)",
        "flags_hdr": "Conditions / Flags",
        "flag_dm": "Diabetes",
        "flag_htn": "Hypertension",
        "flag_dlp": "Dyslipidemia",
        "flag_ob": "Obesity",
        "flag_cort": "High Cortisol",
        "flag_ins": "On Insulin",
        "results_hdr": "Assess",
        "metric_mes": "MES (0–100)",
        "metric_htn": "HTN Risk (0–1)",
        "metric_t2d": "T2D Risk (0–1)",
        "metric_bmi": "BMI",
        "details": "Model Details",
        "recs": "Recommendations",
        "save_btn": "Save Record",
        "saved": "Saved.",
        "history": "History",
        "about": "About",
        "compute": "Compute",
        "download": "Download history (CSV)"
    },
    "ar": {
        "title": "نبأ — النموذج الأولي (كفاءة الأيض وخطر الأمراض)",
        "caption": "عرض تجريبي: يحسب مؤشر كفاءة الأيض MES وقابلية الإصابة بارتفاع الضغط والسكري. ليس للاستخدام الطبي.",
        "lang_label": "اللغة",
        "lang_en": "English",
        "lang_ar": "العربية",
        "inputs_hdr": "المدخلات",
        "age": "العمر",
        "sex": "الجنس",
        "sex_m": "ذكر",
        "sex_f": "أنثى",
        "height": "الطول (سم)",
        "weight": "الوزن (كجم)",
        "steps": "الخطوات (يوميًا)",
        "sleep": "النوم (ساعات)",
        "cal_in": "السعرات الحرارية (كيلو كالوري)",
        "muscle": "نسبة العضل %",
        "sbp": "الضغط الانقباضي (ملم زئبق)",
        "glu": "سكر صائم (ملجم/دل)",
        "sodium": "الصوديوم (ملجم/يوم)",
        "flags_hdr": "الحالات / المؤشرات",
        "flag_dm": "سكري",
        "flag_htn": "ارتفاع ضغط",
        "flag_dlp": "اضطراب دهون",
        "flag_ob": "سمنة",
        "flag_cort": "كورتيزول مرتفع",
        "flag_ins": "يأخذ أنسولين",
        "results_hdr": "التقييم",
        "metric_mes": "MES (0–100)",
        "metric_htn": "خطر الضغط (0–1)",
        "metric_t2d": "خطر السكري (0–1)",
        "metric_bmi": "مؤشر كتلة الجسم",
        "details": "تفاصيل النموذج",
        "recs": "التوصيات",
        "save_btn": "حفظ",
        "saved": "تم الحفظ",
        "history": "السجل",
        "about": "حول",
        "compute": "احسب",
        "download": "تنزيل السجل (CSV)"
    },
}

# ---------------------- Language toggle ----------------------
col_lang_left, col_lang_right = st.columns([1,3])
with col_lang_left:
    language = st.selectbox(
        LANG["en"]["lang_label"] + " / " + LANG["ar"]["lang_label"],
        ["en", "ar"],
        index=0,
        format_func=lambda k: LANG[k]["lang_en"] if k == "en" else LANG[k]["lang_ar"]
    )
L = LANG[language]

# ---------------------- Title ----------------------
st.title(L["title"])
st.caption(L["caption"])

# ---------------------- SQLite (once) ----------------------
conn = sqlite3.connect("naba.db")
cur = conn.cursor()
cur.execute(
    """CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, age INT, sex TEXT, height_cm REAL, weight_kg REAL, steps INT, sleep_hours REAL,
        calories_intake REAL, muscle_percent REAL, bp_systolic REAL, fasting_glucose REAL, sodium_mg REAL,
        flags_json TEXT, mes REAL, htn REAL, t2d REAL
    )"""
)
conn.commit()

# ---------------------- Tabs ----------------------
tab_assess, tab_history, tab_about = st.tabs([L["results_hdr"], L["history"], L["about"]])

# ---------------------- Assess Tab ----------------------
with tab_assess:
    left, right = st.columns([1.2, 1], vertical_alignment="top")

    with left:
        st.subheader(L["inputs_hdr"])
        with st.form("naba_form", clear_on_submit=False):
            colA, colB = st.columns(2)
            with colA:
                age = st.number_input(L["age"], 14, 100, 41, key="age")
                sex_label = st.selectbox(L["sex"], [L["sex_m"], L["sex_f"]], index=0, key="sex_label")
                sex = "M" if sex_label in [LANG["en"]["sex_m"], LANG["ar"]["sex_m"]] else "F"
                height_cm = st.number_input(L["height"], 120.0, 210.0, 175.0, step=0.5, key="height_cm")
                weight_kg = st.number_input(L["weight"], 35.0, 200.0, 95.0, step=0.5, key="weight_kg")
                steps = st.number_input(L["steps"], 0, 40000, 3000, step=100, key="steps")
                sleep_hours = st.number_input(L["sleep"], 0.0, 14.0, 10.0, step=0.25, key="sleep_hours")
            with colB:
                calories_intake = st.number_input(L["cal_in"], 500.0, 6000.0, 3000.0, step=50.0, key="calories_intake")
                muscle_percent = st.number_input(L["muscle"], 5.0, 60.0, 20.0, step=0.5, key="muscle_percent")
                bp_systolic = st.number_input(L["sbp"], 80.0, 220.0, 142.0, step=1.0, key="bp_systolic")
                fasting_glucose = st.number_input(L["glu"], 60.0, 300.0, 110.0, step=1.0, key="fasting_glucose")
                sodium_mg = st.number_input(L["sodium"], 0.0, 8000.0, 3500.0, step=50.0, key="sodium_mg")

            st.markdown(f"**{L['flags_hdr']}**")
            colF1, colF2, colF3 = st.columns(3)
            with colF1:
                f_dm  = st.checkbox(L["flag_dm"], value=True)
                f_htn = st.checkbox(L["flag_htn"], value=True)
            with colF2:
                f_dlp = st.checkbox(L["flag_dlp"], value=True)
                f_ob  = st.checkbox(L["flag_ob"], value=True)
            with colF3:
                f_c   = st.checkbox(L["flag_cort"], value=False)
                f_ins = st.checkbox(L["flag_ins"], value=False)

            submit = st.form_submit_button(L["compute"])

        flags = {
            "diabetes": f_dm, "hypertension": f_htn, "dyslipidemia": f_dlp,
            "obesity": f_ob, "cortisol_high": f_c, "insulin": f_ins
        }

    if submit:
        # compute
        inp = Inputs(
            age=age, sex=sex, height_cm=height_cm, weight_kg=weight_kg, steps=steps, sleep_hours=sleep_hours,
            calories_intake=calories_intake, muscle_percent=muscle_percent, bp_systolic=bp_systolic,
            fasting_glucose=fasting_glucose, sodium_mg=sodium_mg, flags=flags
        )
        mes, info = compute_mes(inp)
        dsi = predict_dsi(inp, mes)
        rec = recommendations(inp, mes, dsi)

        with right:
            st.subheader("Results")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(L["metric_mes"], f"{mes:.0f}")
            c2.metric(L["metric_htn"], f"{dsi['hypertension']:.2f}")
            c3.metric(L["metric_t2d"], f"{dsi['diabetes_t2']:.2f}")
            c4.metric(L["metric_bmi"], f"{bmi(weight_kg, height_cm):.1f}")

            # color cue badges
            risk_htn = "🟢" if dsi["hypertension"] < 0.33 else ("🟠" if dsi["hypertension"] < 0.66 else "🔴")
            risk_dm  = "🟢" if dsi["diabetes_t2"] < 0.33 else ("🟠" if dsi["diabetes_t2"] < 0.66 else "🔴")
            st.info(f"HTN risk {risk_htn} · T2D risk {risk_dm}")

            with st.expander(L["details"], expanded=False):
                st.json({
                    "BMR": round(info["BMR"], 1),
                    "TDEE": round(info["TDEE"], 1),
                    "Intake/TDEE": round(info["Intake/TDEE"], 2),
                    "BaseScore(Energy)": round(info["BaseScore(Energy)"], 1),
                    "Penalty": info["Penalty"],
                })
            with st.expander(L["recs"], expanded=True):
                st.json(rec)

            if st.button(L["save_btn"]):
                cur.execute(
                    "INSERT INTO records (ts, age, sex, height_cm, weight_kg, steps, sleep_hours, calories_intake, muscle_percent, bp_systolic, fasting_glucose, sodium_mg, flags_json, mes, htn, t2d) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (datetime.utcnow().isoformat(), age, sex, height_cm, weight_kg, steps, sleep_hours, calories_intake,
                     muscle_percent, bp_systolic, fasting_glucose, sodium_mg, str(flags), float(mes),
                     float(dsi['hypertension']), float(dsi['diabetes_t2']))
                )
                conn.commit()
                st.success(L["saved"])

# ---------------------- History Tab ----------------------
with tab_history:
    df = pd.read_sql_query(
        "SELECT ts, age, sex, steps, sleep_hours, calories_intake, mes, htn, t2d FROM records ORDER BY id DESC LIMIT 300",
        conn
    )
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(L["download"], data=csv, file_name="naba_history.csv", mime="text/csv")

# ---------------------- About Tab ----------------------
with tab_about:
    st.markdown("""
**NABA** aligns nutrition timing & composition with daily biometrics to reflect your **metabolic rhythm**.
- **MES**: Metabolic Efficiency Score (0–100)
- **DSI**: Risk for Hypertension & Type-2 Diabetes
**Demo only – not medical advice.**
""")

# ---------------------- Footer ----------------------
st.caption("© 2025 NABA — Demo only. Built by Team NABA.")
