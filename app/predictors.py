# app/predictors.py
import os, pickle
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

# ---------- Artifacts ----------
ART_DIR = os.getenv("ART_DIR", "artifacts/v2")
FEATURES_TXT = os.path.join(ART_DIR, "feature_columns.txt")
MODEL_HP = os.path.join(ART_DIR, "model_hypertension.pkl")
MODEL_DM = os.path.join(ART_DIR, "model_diabetes.pkl")
MODEL_DLP = os.path.join(ART_DIR, "model_dyslipidemia.pkl")

# Per-condition decision thresholds (your latest setup)
THRESHOLDS = {"hypertension": 0.50, "diabetes": 0.50, "dyslipidemia": 0.60}

# ---------- Load models & feature schema ----------
with open(FEATURES_TXT, "r", encoding="utf-8") as f:
    FEATURE_COLUMNS = [ln.strip() for ln in f if ln.strip()]

def _load_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)

MODEL = {
    "hypertension": _load_pickle(MODEL_HP),
    "diabetes": _load_pickle(MODEL_DM),
    "dyslipidemia": _load_pickle(MODEL_DLP),
}

# ---------- Feature alignment & simple derives ----------
def _derive(features: Dict[str, Any]) -> Dict[str, Any]:
    # Derive daily averages if yearly totals are present
    out = dict(features)
    steps_y = features.get("steps_year_total")
    cal_y   = features.get("calories_year_total")
    if "steps_day_avg" in FEATURE_COLUMNS and steps_y is not None:
        out["steps_day_avg"] = float(steps_y) / 365.0
    if "cal_day_total" in FEATURE_COLUMNS and cal_y is not None:
        out["cal_day_total"] = float(cal_y) / 365.0
    # MES is provided by a separate function; not a raw input
    return out

def _align_row(features: Dict[str, Any]) -> pd.DataFrame:
    f = _derive(features)
    row = {c: 0.0 for c in FEATURE_COLUMNS}     # missing -> 0
    for k, v in f.items():
        if k in row:
            try:
                row[k] = float(v)
            except Exception:
                row[k] = 0.0
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)

# ---------- MES (0–100; ↑ is better) ----------
# Lightweight scoring consistent with your earlier parts:
GOALS = {
    "a1c": 5.4, "fasting_glucose": 90, "hdl": 60, "tg": 100,
    "systolic": 120, "diastolic": 80, "bmi": 22.5,
    "steps_day_avg": 7000, "cal_day_total": 2000,
}
WEIGHTS = {  # absolute penalty caps per metric
    "a1c": 18, "fasting_glucose": 18, "hdl": 12, "tg": 12,
    "systolic": 10, "diastolic": 8, "bmi": 8,
    "steps_day_avg": 7, "cal_day_total": 5,
    "on_statin": 3, "on_metformin": 3, "on_antihypertensive": 3,
}

def mes_score(features: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    f = _derive(features)
    penalties = {}

    def pen(name, val, goal, mode="lower-better", scale=1.0, cap=10):
        if val is None: return 0.0
        d = 0.0
        if mode == "lower-better":
            d = max(0.0, (float(val) - goal) / scale)
        elif mode == "higher-better":
            d = max(0.0, (goal - float(val)) / scale)
        return min(d, cap)

    penalties["a1c"]             = pen("a1c", f.get("a1c"), GOALS["a1c"], "lower-better", 0.3, WEIGHTS["a1c"])
    penalties["fasting_glucose"] = pen("fasting_glucose", f.get("fasting_glucose"), GOALS["fasting_glucose"], "lower-better", 5.0, WEIGHTS["fasting_glucose"])
    penalties["hdl"]             = pen("hdl", f.get("hdl"), GOALS["hdl"], "higher-better", 2.0, WEIGHTS["hdl"])
    penalties["tg"]              = pen("tg", f.get("tg"), GOALS["tg"], "lower-better", 15.0, WEIGHTS["tg"])
    penalties["systolic"]        = pen("systolic", f.get("systolic"), GOALS["systolic"], "lower-better", 4.0, WEIGHTS["systolic"])
    penalties["diastolic"]       = pen("diastolic", f.get("diastolic"), GOALS["diastolic"], "lower-better", 3.0, WEIGHTS["diastolic"])
    penalties["bmi"]             = pen("bmi", f.get("bmi"), GOALS["bmi"], "lower-better", 1.0, WEIGHTS["bmi"])
    penalties["steps_day_avg"]   = pen("steps_day_avg", f.get("steps_day_avg"), GOALS["steps_day_avg"], "higher-better", 1000.0, WEIGHTS["steps_day_avg"])
    penalties["cal_day_total"]   = pen("cal_day_total", f.get("cal_day_total"), GOALS["cal_day_total"], "lower-better", 200.0, WEIGHTS["cal_day_total"])
    # meds flags (penalize slightly if on chronic meds)
    for m in ("on_statin","on_metformin","on_antihypertensive"):
        penalties[m] = WEIGHTS[m] if float(f.get(m,0)) > 0 else 0.0

    total_pen = sum(penalties.values())
    mes = max(0.0, 100.0 - total_pen)  # 0–100
    return round(mes, 1), penalties

# ---------- Predict ----------
def predict_all(features: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    x = _align_row(features)
    out = {}
    for cond, mdl in MODEL.items():
        prob = float(mdl.predict_proba(x)[0, 1])
        thr = THRESHOLDS[cond]
        out[cond] = {"prob": prob, "label": int(prob >= thr), "thr": thr}
    return out
