import streamlit as st
import sqlite3
from datetime import datetime
import pandas as pd
from naba_core import Inputs, compute_mes, predict_dsi, recommendations, bmi

# ---------- Language pack ----------
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
        "results_hdr": "Results",
        "metric_mes": "MES (0–100)",
        "metric_htn": "HTN Risk (0–1)",
        "metric_t2d": "T2D Risk (0–1)",
        "metric_bmi": "BMI",
        "details": "Model Details",
        "recs": "Recommendations",
        "save_btn": "Save Record",
        "saved": "Saved.",
        "history": "History",
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
        "results_hdr": "النتائج",
        "metric_mes": "MES (0–100)",
        "metric_htn": "خطر الضغط (0–1)",
        "metric_t2d": "خطر السكري (0–1)",
        "metric_bmi": "مؤشر كتلة الجسم",
        "details": "تفاصيل النموذج",
        "recs": "التوصيات",
        "save_btn": "حفظ",
        "saved": "تم الحفظ",
        "history": "السجل",
    },
}

st.set_page_config(page_title="NABA MVP", page_icon="🧬", layout="wide")

# -------- Language toggle (top bar) --------
col_lang_left, col_lang_right = st.columns([1,3])
with col_lang_left:
    language = st.selectbox(LANG["en"]["lang_label"] + " / " + LANG["ar"]["lang_label"],
                            ["en", "ar"], index=0, format_func=lambda k: LANG[k]["lang_en"] if k=="en" else LANG[k]["lang_ar"])
L = LANG[language]

# -------- Title & caption --------
st.title(L["title"])
st.caption(L["caption"])

# -------- SQLite --------
conn = sqlite3.connect("naba.db")
cur = conn.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, age INT, sex TEXT, height_cm REAL, weight_kg REAL, steps INT, sleep_hours REAL,
    calories_intake REAL, muscle_percent REAL, bp_systolic REAL, fasting_glucose REAL, sodium_mg REAL,
    flags_json TEXT, mes REAL, htn REAL, t2d REAL
)""")
conn.commit()

# -------- Sidebar Inputs --------
with st.sidebar:
    st.header(L["inputs_hdr"])
    colA, colB = st.columns(2)
    with colA:
        age = st.number_input(L["age"], 14, 100, 41)
        sex_label = st.selectbox(L["sex"], [L["sex_m"], L["sex_f"]], index=0)
        sex = "M" if sex_label in [LANG["en"]["sex_m"], LANG["ar"]["sex_m"]] else "F"
        height_cm = st.number_input(L["height"], 120.0, 210.0, 175.0, step=0.5)
        weight_kg = st.number_input(L["weight"], 35.0, 200.0, 95.0, step=0.5)
        steps = st.number_input(L["steps"], 0, 40000, 3000, step=100)
        sleep_hours = st.number_input(L["sleep"], 0.0, 14.0, 10.0, step=0.25)
    with colB:
        calories_intake = st.number_input(L["cal_in"], 500.0, 6000.0, 3000.0, step=50.0)
        muscle_percent = st.number_input(L["muscle"], 5.0, 60.0, 20.0, step=0.5)
        bp_systolic = st.number_input(L["sbp"], 80.0, 220.0, 142.0, step=1.0)
        fasting_glucose = st.number_input(L["glu"], 60.0, 300.0, 110.0, step=1.0)
        sodium_mg = st.number_input(L["sodium"], 0.0, 8000.0, 3500.0, step=50.0)

    st.markdown(f"**{L['flags_hdr']}**")
    flags = {
        "diabetes": st.checkbox(L["flag_dm"], value=True),
        "hypertension": st.checkbox(L["flag_htn"], value=True),
        "dyslipidemia": st.checkbox(L["flag_dlp"], value=True),
        "obesity": st.checkbox(L["flag_ob"], value=True),
        "cortisol_high": st.checkbox(L["flag_cort"], value=False),
        "insulin": st.checkbox(L["flag_ins"], value=False),
    }

# -------- Compute --------
st.subheader(L["results_hdr"])
inp = Inputs(
    age=age, sex=sex, height_cm=height_cm, weight_kg=weight_kg, steps=steps, sleep_hours=sleep_hours,
    calories_intake=calories_intake, muscle_percent=muscle_percent, bp_systolic=bp_systolic,
    fasting_glucose=fasting_glucose, sodium_mg=sodium_mg, flags=flags
)
mes, info = compute_mes(inp)
dsi = predict_dsi(inp, mes)
rec = recommendations(inp, mes, dsi)

col1, col2, col3, col4 = st.columns(4)
col1.metric(L["metric_mes"], f"{mes:.0f}")
col2.metric(L["metric_htn"], f"{dsi['hypertension']:.2f}")
col3.metric(L["metric_t2d"], f"{dsi['diabetes_t2']:.2f}")
col4.metric(L["metric_bmi"], f"{bmi(weight_kg, height_cm):.1f}")

with st.expander(L["details"]):
    st.json({
        "BMR": round(info["BMR"], 1),
        "TDEE": round(info["TDEE"], 1),
        "Intake/TDEE": round(info["Intake/TDEE"], 2),
        "BaseScore(Energy)": round(info["BaseScore(Energy)"], 1),
        "Penalty": info["Penalty"],
    })

with st.expander(L["recs"]):
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

st.subheader(L["history"])
df = pd.read_sql_query(
    "SELECT ts, age, sex, steps, sleep_hours, calories_intake, mes, htn, t2d FROM records ORDER BY id DESC LIMIT 100",
    conn
)
st.dataframe(df, use_container_width=True)
