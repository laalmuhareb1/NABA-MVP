from dataclasses import dataclass
from typing import Dict, Tuple
import math

@dataclass
class Inputs:
    age: int
    sex: str
    height_cm: float
    weight_kg: float
    steps: int
    sleep_hours: float
    calories_intake: float
    muscle_percent: float
    bp_systolic: float
    fasting_glucose: float
    sodium_mg: float
    flags: Dict[str, int]

def bmi(weight_kg: float, height_cm: float) -> float:
    if height_cm <= 0: return 0.0
    return weight_kg / ((height_cm/100.0)**2)

def mifflin_st_jeor(sex: str, weight_kg: float, height_cm: float, age: int) -> float:
    s = 5 if sex.upper() == 'M' else -161
    return 10*weight_kg + 6.25*height_cm - 5*age + s

def sleep_modifier(sleep_h: float) -> float:
    if sleep_h < 6: return 0.92
    if sleep_h < 7: return 0.97
    if sleep_h <= 8: return 1.00
    if sleep_h <= 9: return 0.98
    if sleep_h <= 10: return 0.96
    return 0.95

def steps_to_kcal(steps: int, weight_kg: float) -> float:
    return max(0.0, 0.045 * float(steps))

def compute_tdee(inp: Inputs) -> float:
    bmr = mifflin_st_jeor(inp.sex, inp.weight_kg, inp.height_cm, inp.age)
    activity_kcal = steps_to_kcal(inp.steps, inp.weight_kg)
    tdee = bmr * 1.2 + activity_kcal
    tdee *= sleep_modifier(inp.sleep_hours)
    if inp.sex.upper() == 'M' and inp.muscle_percent < 30:
        tdee *= 0.97
    return max(1.0, tdee)

def compute_mes(inp: Inputs) -> Tuple[float, dict]:
    tdee = compute_tdee(inp)
    ratio = inp.calories_intake / tdee
    base = 100 - min(abs(1.0 - ratio) * 100, 60)
    penalty = 0.0
    penalty += 6 if inp.flags.get('diabetes',0) else 0
    penalty += 4 if inp.flags.get('obesity',0) else 0
    penalty += 3 if inp.flags.get('hypertension',0) else 0
    penalty += 3 if inp.flags.get('dyslipidemia',0) else 0
    penalty += 2 if inp.flags.get('insulin',0) else 0
    penalty += 2 if inp.flags.get('cortisol_high',0) else 0
    if inp.steps < 5000: penalty += 3
    if inp.sleep_hours >= 9.5 or inp.sleep_hours < 6: penalty += 1
    if inp.muscle_percent < 25: penalty += 2
    score = max(0.0, min(100.0, base - penalty))
    details = {
        'BMR': mifflin_st_jeor(inp.sex, inp.weight_kg, inp.height_cm, inp.age),
        'TDEE': tdee,
        'Intake/TDEE': ratio,
        'BaseScore(Energy)': base,
        'Penalty': penalty
    }
    return score, details

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def predict_dsi(inp: Inputs, mes: float) -> dict:
    z_htn = -2.0 + 0.0005*inp.sodium_mg + 0.02*(140 - mes) + 0.002*max(0, 140 - inp.bp_systolic) + (-0.0002)*inp.steps
    z_dm  = -1.6 + 0.01*max(0, inp.fasting_glucose - 100) + 0.02*(90 - mes) + (-0.00015)*inp.steps
    return {
        'hypertension': max(0.0, min(1.0, sigmoid(z_htn))),
        'diabetes_t2': max(0.0, min(1.0, sigmoid(z_dm)))
    }

def protein_target(weight_kg: float, muscle_pct: float) -> float:
    lean_mass = weight_kg * (muscle_pct/100.0)
    return max(1.2*weight_kg, 1.6*lean_mass)

def recommendations(inp: Inputs, mes: float, dsi: dict) -> dict:
    rec = {}
    ratio = inp.calories_intake / compute_tdee(inp)
    if ratio > 1.1:
        rec['Energy'] = 'Reduce energy intake by 10–15% or add 2–3k steps/day.'
    elif ratio < 0.9:
        rec['Energy'] = 'Increase intake ~10% with nutrient-dense foods.'
    else:
        rec['Energy'] = 'Energy balance near optimal window.'
    rec['Protein'] = f'Target protein ≈ {protein_target(inp.weight_kg, inp.muscle_percent):.0f} g/day.'
    if dsi['hypertension'] >= 0.5:
        rec['Sodium'] = 'Keep sodium < 2 g/day; increase potassium (leafy greens, legumes).'
    if dsi['diabetes_t2'] >= 0.5 or inp.flags.get('diabetes',0):
        rec['Timing'] = 'Front-load carbs earlier; last meal ≥4h before sleep; prioritize protein/veg at dinner.'
    if inp.steps < 7000:
        rec['Activity'] = 'Add 10–15 min post-meal walks (2–3x/day); aim ≥7,000–8,500 steps/day.'
    if inp.sleep_hours >= 9.5 or inp.sleep_hours < 6:
        rec['Sleep'] = 'Consolidate to 7–8 h/night with fixed wake time.'
    rec['Summary'] = f\"MES={mes:.0f}/100, HTN risk={dsi['hypertension']:.2f}, T2D risk={dsi['diabetes_t2']:.2f}.\"
    return rec
