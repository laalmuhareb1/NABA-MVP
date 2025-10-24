# NABA — MVP (Bilingual EN/AR)

This MVP computes:
- **Metabolic Efficiency Score (MES)** (0–100)
- **Disease Susceptibility (DSI)** for **Hypertension** & **Type 2 Diabetes**
- Shows **recommendations** and **saves history** (SQLite)

**Languages:** English / العربية (toggle in the app)

## How to run locally
```bash
python3 -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
