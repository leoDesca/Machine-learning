"""
FastAPI Deployment Server — Makerere Food Supply Chain Forecasting
Serves XGBoost predictions + a beautiful web dashboard UI.

RUN:
    python -m uvicorn app:app --reload --port 8000

Then open: http://127.0.0.1:8000
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import os
from datetime import date, timedelta

# ── App init ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Makerere Food Supply Chain — Demand Forecasting API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models ────────────────────────────────────────────────────────────────
MODEL_DIR = "models"

cafe_model  = xgb.XGBRegressor()
kiosk_model = xgb.XGBRegressor()

cafe_model.load_model(os.path.join(MODEL_DIR, "xgb_cafeteria.json"))
kiosk_model.load_model(os.path.join(MODEL_DIR, "xgb_kiosk.json"))

with open(os.path.join(MODEL_DIR, "cafe_features.pkl"),  "rb") as f:
    CAFE_FEATURES  = pickle.load(f)
with open(os.path.join(MODEL_DIR, "kiosk_features.pkl"), "rb") as f:
    KIOSK_FEATURES = pickle.load(f)

print("✅ Models loaded successfully")

PERIOD_MAP = {
    "Sem1_Teaching": 1, "Sem1_Exams": 2, "Sem1_Break": 3,
    "Sem2_Teaching": 4, "Sem2_Exams": 5, "Sem2_Break": 6
}

# ── Schemas ────────────────────────────────────────────────────────────────────
class CafeteriaRequest(BaseModel):
    prediction_date: date
    lag_1: float;  lag_2: float;  lag_3: float
    lag_7: float;  lag_14: float; lag_21: float
    roll_mean_7: float;  roll_mean_14: float; roll_mean_21: float
    roll_std_7: float;   roll_std_14: float;  roll_std_21: float
    roll_max_7: float;   roll_max_14: float;  roll_max_21: float
    roll_min_7: float;   roll_min_14: float;  roll_min_21: float
    portions_sold: float; portions_prepared: float
    waste_portions: float = 0; waste_pct: float = 0
    ingredient_cost_ugx: float
    is_weekend: int = Field(ge=0, le=1)
    academic_period: str
    profit_margin: float = 0.3
    waste_ratio: float = 0.05

class KioskRequest(BaseModel):
    prediction_date: date
    lag_1: float;  lag_2: float;  lag_3: float
    lag_7: float;  lag_14: float; lag_21: float
    roll_mean_7: float;  roll_mean_14: float; roll_mean_21: float
    roll_std_7: float;   roll_std_14: float;  roll_std_21: float
    roll_max_7: float;   roll_max_14: float;  roll_max_21: float
    roll_min_7: float;   roll_min_14: float;  roll_min_21: float
    num_transactions: float; avg_unit_price: float; total_quantity: float
    is_weekend: int = Field(ge=0, le=1)
    near_lecture_start: float = 0
    academic_period: str
    avg_transaction_value: float = 0

class PredictionResponse(BaseModel):
    prediction_date: str
    predicted_revenue: float
    predicted_revenue_M: str
    model: str
    confidence_note: str

# ── Helpers ────────────────────────────────────────────────────────────────────
def date_to_calendar(d: date) -> dict:
    return {
        "day_of_week":  d.weekday(),
        "month":        d.month,
        "week_of_year": d.isocalendar()[1],
        "quarter":      (d.month - 1) // 3 + 1,
        "day_of_month": d.day,
        "is_month_end": int(d == (d.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)),
    }

def build_row(req_dict: dict, features: list) -> pd.DataFrame:
    d = date.fromisoformat(str(req_dict.pop("prediction_date")))
    cal = date_to_calendar(d)
    rename = {
        "portions_sold": "Portions_Sold", "portions_prepared": "Portions_Prepared",
        "waste_portions": "Waste_Portions", "waste_pct": "Waste_Pct",
        "ingredient_cost_ugx": "Ingredient_Cost_UGX", "is_weekend": "Is_Weekend",
        "num_transactions": "Num_Transactions", "avg_unit_price": "Avg_Unit_Price",
        "total_quantity": "Total_Quantity", "near_lecture_start": "Near_Lecture_Start",
        "profit_margin": "profit_margin", "waste_ratio": "waste_ratio",
        "avg_transaction_value": "avg_transaction_value",
    }
    row = {rename.get(k, k): v for k, v in req_dict.items()}
    if "academic_period" in row:
        row["Academic_Period_num"] = PERIOD_MAP.get(row.pop("academic_period"), 0)
    row.update(cal)
    df = pd.DataFrame([row])
    for feat in features:
        if feat not in df.columns:
            df[feat] = 0.0
    return df[features]

# ── API Routes ─────────────────────────────────────────────────────────────────
@app.post("/predict/cafeteria", response_model=PredictionResponse)
def predict_cafeteria(req: CafeteriaRequest):
    try:
        row  = build_row(req.dict(), CAFE_FEATURES)
        pred = max(float(cafe_model.predict(row)[0]), 0)
        return PredictionResponse(
            prediction_date=str(req.prediction_date),
            predicted_revenue=round(pred, 2),
            predicted_revenue_M=f"{pred/1_000_000:.2f}M UGX",
            model="XGBoost-Cafeteria-v1",
            confidence_note="Based on historical demand patterns and academic calendar.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/kiosk", response_model=PredictionResponse)
def predict_kiosk(req: KioskRequest):
    try:
        row  = build_row(req.dict(), KIOSK_FEATURES)
        pred = max(float(kiosk_model.predict(row)[0]), 0)
        return PredictionResponse(
            prediction_date=str(req.prediction_date),
            predicted_revenue=round(pred, 2),
            predicted_revenue_M=f"{pred/1_000_000:.2f}M UGX",
            model="XGBoost-Kiosk-v1",
            confidence_note="Based on transaction patterns and lecture schedule.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
def api_status():
    return {"status": "online", "models": ["cafeteria", "kiosk"], "version": "1.0.0"}

# ── Dashboard HTML ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Makerere Demand Forecasting</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #0a0e17;
    --surface:  #111827;
    --surface2: #1a2235;
    --border:   rgba(255,255,255,0.07);
    --border2:  rgba(255,255,255,0.13);
    --text:     #e8eaf0;
    --muted:    #6b7a99;
    --accent:   #00d4aa;
    --accent2:  #0099ff;
    --warn:     #f59e0b;
    --cafe:     #00d4aa;
    --kiosk:    #6366f1;
    --radius:   12px;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html { scroll-behavior: smooth; }

  body {
    font-family: 'DM Mono', monospace;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Grid background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(0,212,170,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,212,170,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  /* ── Header ── */
  header {
    position: relative;
    z-index: 10;
    padding: 28px 48px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    backdrop-filter: blur(12px);
    background: rgba(10,14,23,0.8);
    position: sticky;
    top: 0;
  }
  .logo {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .logo-mark {
    width: 38px; height: 38px;
    border: 1.5px solid var(--accent);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 14px;
    color: var(--accent);
    letter-spacing: 0.5px;
  }
  .logo-text {
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: var(--text);
    letter-spacing: 0.3px;
    line-height: 1.3;
  }
  .logo-sub {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    font-weight: 300;
    color: var(--muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
  }
  .status-pill {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 14px;
    border: 1px solid rgba(0,212,170,0.3);
    border-radius: 20px;
    font-size: 11px;
    color: var(--accent);
    letter-spacing: 1px;
    text-transform: uppercase;
  }
  .status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.5; transform: scale(0.8); }
  }

  /* ── Layout ── */
  main {
    position: relative;
    z-index: 1;
    max-width: 1100px;
    margin: 0 auto;
    padding: 48px 32px 80px;
  }

  /* ── Hero ── */
  .hero {
    text-align: center;
    margin-bottom: 56px;
    animation: fadeUp 0.6s ease both;
  }
  .hero-eyebrow {
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 16px;
  }
  .hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(32px, 5vw, 52px);
    font-weight: 800;
    line-height: 1.1;
    color: var(--text);
    margin-bottom: 16px;
  }
  .hero h1 span { color: var(--accent); }
  .hero p {
    font-size: 14px;
    color: var(--muted);
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.8;
  }

  /* ── Stats row ── */
  .stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 48px;
    animation: fadeUp 0.6s 0.1s ease both;
  }
  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 24px;
    transition: border-color 0.2s;
  }
  .stat-card:hover { border-color: var(--border2); }
  .stat-label {
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
  }
  .stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 700;
    color: var(--text);
  }
  .stat-value.green { color: var(--cafe); }
  .stat-value.blue  { color: var(--kiosk); }
  .stat-note { font-size: 11px; color: var(--muted); margin-top: 4px; }

  /* ── Tabs ── */
  .tabs {
    display: flex;
    gap: 4px;
    margin-bottom: 24px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 5px;
    width: fit-content;
    animation: fadeUp 0.6s 0.2s ease both;
  }
  .tab {
    padding: 9px 24px;
    border-radius: 8px;
    border: none;
    background: transparent;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.5px;
    cursor: pointer;
    transition: all 0.2s;
  }
  .tab.active {
    background: var(--surface2);
    color: var(--text);
    border: 1px solid var(--border2);
  }
  .tab:hover:not(.active) { color: var(--text); }

  /* ── Form panels ── */
  .panel { display: none; animation: fadeUp 0.4s ease both; }
  .panel.active { display: block; }

  .form-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px;
  }
  .card-title {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
  }

  .field { margin-bottom: 16px; }
  label {
    display: block;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
  }
  input, select {
    width: 100%;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    padding: 10px 14px;
    outline: none;
    transition: border-color 0.2s;
    appearance: none;
  }
  input:focus, select:focus { border-color: var(--accent); }
  input::placeholder { color: var(--muted); }
  select { cursor: pointer; }
  select option { background: var(--surface); }

  .field-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 10px;
  }
  .field-row.two { grid-template-columns: 1fr 1fr; }

  /* ── Submit button ── */
  .btn-predict {
    width: 100%;
    padding: 14px;
    margin-top: 8px;
    background: var(--accent);
    color: #0a0e17;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.5px;
    cursor: pointer;
    transition: all 0.2s;
    position: relative;
    overflow: hidden;
  }
  .btn-predict::after {
    content: '';
    position: absolute;
    inset: 0;
    background: white;
    opacity: 0;
    transition: opacity 0.2s;
  }
  .btn-predict:hover::after { opacity: 0.1; }
  .btn-predict:active { transform: scale(0.98); }
  .btn-predict.kiosk { background: var(--kiosk); color: white; }
  .btn-predict:disabled { opacity: 0.5; cursor: not-allowed; }

  /* ── Result card ── */
  .result-card {
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    padding: 28px;
    display: none;
    animation: fadeUp 0.4s ease both;
  }
  .result-card.visible { display: block; }
  .result-label {
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
  }
  .result-amount {
    font-family: 'Syne', sans-serif;
    font-size: 48px;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
    margin-bottom: 8px;
  }
  .result-amount.kiosk { color: var(--kiosk); }
  .result-sub {
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 20px;
  }
  .result-meta {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
  }
  .meta-row {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
  }
  .meta-key { color: var(--muted); }
  .meta-val { color: var(--text); }

  .error-box {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 12px;
    color: #f87171;
    display: none;
    margin-top: 12px;
  }
  .error-box.visible { display: block; }

  .loading-bar {
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 2px;
    width: 0%;
    transition: width 0.3s ease;
    margin-top: 12px;
  }

  /* ── Footer ── */
  footer {
    position: relative;
    z-index: 1;
    text-align: center;
    padding: 32px;
    border-top: 1px solid var(--border);
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.5px;
  }
  footer a { color: var(--accent); text-decoration: none; }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  @media (max-width: 768px) {
    header { padding: 20px 24px; }
    main { padding: 32px 20px 60px; }
    .form-grid { grid-template-columns: 1fr; }
    .stats { grid-template-columns: 1fr; }
    .field-row { grid-template-columns: 1fr 1fr; }
    .hero h1 { font-size: 28px; }
  }
</style>
</head>
<body>

<!-- Header -->
<header>
  <div class="logo">
    <div class="logo-mark">MU</div>
    <div>
      <div class="logo-text">Demand Forecasting</div>
      <div class="logo-sub">Makerere University</div>
    </div>
  </div>
  <div class="status-pill">
    <div class="status-dot"></div>
    Models Online
  </div>
</header>

<!-- Main -->
<main>

  <!-- Hero -->
  <div class="hero">
    <div class="hero-eyebrow">XGBoost · Supply Chain Intelligence</div>
    <h1>Predict <span>Revenue</span><br>Before the Day Starts</h1>
    <p>Enter recent sales history and context to get an instant XGBoost forecast for cafeteria or kiosk daily revenue.</p>
  </div>

  <!-- Stats -->
  <div class="stats">
    <div class="stat-card">
      <div class="stat-label">Model Type</div>
      <div class="stat-value green">XGBoost</div>
      <div class="stat-note">Gradient boosted trees</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Training Data</div>
      <div class="stat-value">700</div>
      <div class="stat-note">Days of historical data</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Outlets</div>
      <div class="stat-value blue">2</div>
      <div class="stat-note">Cafeteria + Kiosk</div>
    </div>
  </div>

  <!-- Tabs -->
  <div class="tabs">
    <button class="tab active" onclick="switchTab('cafe')">Cafeteria</button>
    <button class="tab" onclick="switchTab('kiosk')">Kiosk</button>
  </div>

  <!-- Cafeteria Panel -->
  <div class="panel active" id="panel-cafe">
    <div class="form-grid">
      <div>
        <!-- Lag features -->
        <div class="card" style="margin-bottom:20px">
          <div class="card-title">Recent Revenue (UGX)</div>
          <div class="field-row">
            <div class="field"><label>Yesterday</label><input type="number" id="c_lag1" placeholder="4200000" value="4200000"></div>
            <div class="field"><label>2 days ago</label><input type="number" id="c_lag2" placeholder="3900000" value="3900000"></div>
            <div class="field"><label>3 days ago</label><input type="number" id="c_lag3" placeholder="4100000" value="4100000"></div>
          </div>
          <div class="field-row">
            <div class="field"><label>7 days ago</label><input type="number" id="c_lag7" placeholder="3800000" value="3800000"></div>
            <div class="field"><label>14 days ago</label><input type="number" id="c_lag14" placeholder="4000000" value="4000000"></div>
            <div class="field"><label>21 days ago</label><input type="number" id="c_lag21" placeholder="3700000" value="3700000"></div>
          </div>
        </div>

        <!-- Rolling stats -->
        <div class="card" style="margin-bottom:20px">
          <div class="card-title">Rolling Statistics</div>
          <div class="field-row">
            <div class="field"><label>7-day mean</label><input type="number" id="c_rm7" placeholder="4000000" value="4000000"></div>
            <div class="field"><label>14-day mean</label><input type="number" id="c_rm14" placeholder="3950000" value="3950000"></div>
            <div class="field"><label>21-day mean</label><input type="number" id="c_rm21" placeholder="3900000" value="3900000"></div>
          </div>
          <div class="field-row">
            <div class="field"><label>7-day std</label><input type="number" id="c_rs7" placeholder="150000" value="150000"></div>
            <div class="field"><label>7-day max</label><input type="number" id="c_rmax7" placeholder="4500000" value="4500000"></div>
            <div class="field"><label>7-day min</label><input type="number" id="c_rmin7" placeholder="3500000" value="3500000"></div>
          </div>
          <div class="field-row">
            <div class="field"><label>14-day std</label><input type="number" id="c_rs14" placeholder="160000" value="160000"></div>
            <div class="field"><label>14-day max</label><input type="number" id="c_rmax14" placeholder="4600000" value="4600000"></div>
            <div class="field"><label>14-day min</label><input type="number" id="c_rmin14" placeholder="3400000" value="3400000"></div>
          </div>
          <div class="field-row">
            <div class="field"><label>21-day std</label><input type="number" id="c_rs21" placeholder="170000" value="170000"></div>
            <div class="field"><label>21-day max</label><input type="number" id="c_rmax21" placeholder="4700000" value="4700000"></div>
            <div class="field"><label>21-day min</label><input type="number" id="c_rmin21" placeholder="3300000" value="3300000"></div>
          </div>
        </div>
      </div>

      <div>
        <!-- Context -->
        <div class="card" style="margin-bottom:20px">
          <div class="card-title">Day Context</div>
          <div class="field">
            <label>Prediction Date</label>
            <input type="date" id="c_date">
          </div>
          <div class="field-row two">
            <div class="field">
              <label>Day type</label>
              <select id="c_weekend">
                <option value="0">Weekday</option>
                <option value="1">Weekend</option>
              </select>
            </div>
            <div class="field">
              <label>Academic period</label>
              <select id="c_period">
                <option value="Sem1_Teaching">Sem 1 Teaching</option>
                <option value="Sem1_Exams">Sem 1 Exams</option>
                <option value="Sem1_Break">Sem 1 Break</option>
                <option value="Sem2_Teaching">Sem 2 Teaching</option>
                <option value="Sem2_Exams">Sem 2 Exams</option>
                <option value="Sem2_Break">Sem 2 Break</option>
              </select>
            </div>
          </div>
          <div class="field-row two">
            <div class="field"><label>Portions sold (est.)</label><input type="number" id="c_psold" placeholder="320" value="320"></div>
            <div class="field"><label>Portions prepared</label><input type="number" id="c_pprep" placeholder="350" value="350"></div>
          </div>
          <div class="field-row two">
            <div class="field"><label>Ingredient cost (UGX)</label><input type="number" id="c_cost" placeholder="1200000" value="1200000"></div>
            <div class="field"><label>Profit margin (0–1)</label><input type="number" id="c_margin" step="0.01" placeholder="0.30" value="0.30"></div>
          </div>
        </div>

        <!-- Result -->
        <div class="result-card" id="c_result">
          <div class="result-label">Predicted Revenue</div>
          <div class="result-amount" id="c_amount">—</div>
          <div class="result-sub" id="c_sub">—</div>
          <div class="result-meta">
            <div class="meta-row"><span class="meta-key">Date</span><span class="meta-val" id="c_rdate">—</span></div>
            <div class="meta-row"><span class="meta-key">Model</span><span class="meta-val">XGBoost-Cafeteria-v1</span></div>
            <div class="meta-row"><span class="meta-key">Status</span><span class="meta-val" style="color:var(--cafe)">Prediction complete</span></div>
          </div>
        </div>
        <div class="error-box" id="c_error"></div>
        <div class="loading-bar" id="c_bar"></div>

        <button class="btn-predict" id="c_btn" onclick="predictCafe()">
          Run Cafeteria Forecast
        </button>
      </div>
    </div>
  </div>

  <!-- Kiosk Panel -->
  <div class="panel" id="panel-kiosk">
    <div class="form-grid">
      <div>
        <div class="card" style="margin-bottom:20px">
          <div class="card-title">Recent Revenue (UGX)</div>
          <div class="field-row">
            <div class="field"><label>Yesterday</label><input type="number" id="k_lag1" placeholder="800000" value="800000"></div>
            <div class="field"><label>2 days ago</label><input type="number" id="k_lag2" placeholder="750000" value="750000"></div>
            <div class="field"><label>3 days ago</label><input type="number" id="k_lag3" placeholder="820000" value="820000"></div>
          </div>
          <div class="field-row">
            <div class="field"><label>7 days ago</label><input type="number" id="k_lag7" placeholder="780000" value="780000"></div>
            <div class="field"><label>14 days ago</label><input type="number" id="k_lag14" placeholder="760000" value="760000"></div>
            <div class="field"><label>21 days ago</label><input type="number" id="k_lag21" placeholder="740000" value="740000"></div>
          </div>
        </div>

        <div class="card" style="margin-bottom:20px">
          <div class="card-title">Rolling Statistics</div>
          <div class="field-row">
            <div class="field"><label>7-day mean</label><input type="number" id="k_rm7" placeholder="780000" value="780000"></div>
            <div class="field"><label>14-day mean</label><input type="number" id="k_rm14" placeholder="770000" value="770000"></div>
            <div class="field"><label>21-day mean</label><input type="number" id="k_rm21" placeholder="760000" value="760000"></div>
          </div>
          <div class="field-row">
            <div class="field"><label>7-day std</label><input type="number" id="k_rs7" placeholder="40000" value="40000"></div>
            <div class="field"><label>7-day max</label><input type="number" id="k_rmax7" placeholder="900000" value="900000"></div>
            <div class="field"><label>7-day min</label><input type="number" id="k_rmin7" placeholder="680000" value="680000"></div>
          </div>
          <div class="field-row">
            <div class="field"><label>14-day std</label><input type="number" id="k_rs14" placeholder="45000" value="45000"></div>
            <div class="field"><label>14-day max</label><input type="number" id="k_rmax14" placeholder="920000" value="920000"></div>
            <div class="field"><label>14-day min</label><input type="number" id="k_rmin14" placeholder="660000" value="660000"></div>
          </div>
          <div class="field-row">
            <div class="field"><label>21-day std</label><input type="number" id="k_rs21" placeholder="50000" value="50000"></div>
            <div class="field"><label>21-day max</label><input type="number" id="k_rmax21" placeholder="940000" value="940000"></div>
            <div class="field"><label>21-day min</label><input type="number" id="k_rmin21" placeholder="640000" value="640000"></div>
          </div>
        </div>
      </div>

      <div>
        <div class="card" style="margin-bottom:20px">
          <div class="card-title">Day Context</div>
          <div class="field">
            <label>Prediction Date</label>
            <input type="date" id="k_date">
          </div>
          <div class="field-row two">
            <div class="field">
              <label>Day type</label>
              <select id="k_weekend">
                <option value="0">Weekday</option>
                <option value="1">Weekend</option>
              </select>
            </div>
            <div class="field">
              <label>Academic period</label>
              <select id="k_period">
                <option value="Sem1_Teaching">Sem 1 Teaching</option>
                <option value="Sem1_Exams">Sem 1 Exams</option>
                <option value="Sem1_Break">Sem 1 Break</option>
                <option value="Sem2_Teaching">Sem 2 Teaching</option>
                <option value="Sem2_Exams">Sem 2 Exams</option>
                <option value="Sem2_Break">Sem 2 Break</option>
              </select>
            </div>
          </div>
          <div class="field-row two">
            <div class="field"><label>Num transactions</label><input type="number" id="k_txn" placeholder="95" value="95"></div>
            <div class="field"><label>Avg unit price (UGX)</label><input type="number" id="k_price" placeholder="8500" value="8500"></div>
          </div>
          <div class="field-row two">
            <div class="field"><label>Total quantity</label><input type="number" id="k_qty" placeholder="110" value="110"></div>
            <div class="field"><label>Near lecture start</label><input type="number" id="k_lecture" step="0.1" placeholder="0.4" value="0.4"></div>
          </div>
        </div>

        <div class="result-card" id="k_result">
          <div class="result-label">Predicted Revenue</div>
          <div class="result-amount kiosk" id="k_amount">—</div>
          <div class="result-sub" id="k_sub">—</div>
          <div class="result-meta">
            <div class="meta-row"><span class="meta-key">Date</span><span class="meta-val" id="k_rdate">—</span></div>
            <div class="meta-row"><span class="meta-key">Model</span><span class="meta-val">XGBoost-Kiosk-v1</span></div>
            <div class="meta-row"><span class="meta-key">Status</span><span class="meta-val" style="color:var(--kiosk)">Prediction complete</span></div>
          </div>
        </div>
        <div class="error-box" id="k_error"></div>
        <div class="loading-bar" id="k_bar"></div>

        <button class="btn-predict kiosk" id="k_btn" onclick="predictKiosk()">
          Run Kiosk Forecast
        </button>
      </div>
    </div>
  </div>

</main>

<footer>
  Makerere University · Food Supply Chain Intelligence ·
  <a href="/docs">API Docs</a> · XGBoost v2
</footer>

<script>
  // Set today's date as default
  const today = new Date().toISOString().split('T')[0];
  document.getElementById('c_date').value = today;
  document.getElementById('k_date').value = today;

  function switchTab(tab) {
    document.querySelectorAll('.tab').forEach((t,i) => t.classList.toggle('active', (tab==='cafe'&&i===0)||(tab==='kiosk'&&i===1)));
    document.getElementById('panel-cafe').classList.toggle('active', tab==='cafe');
    document.getElementById('panel-kiosk').classList.toggle('active', tab==='kiosk');
  }

  function v(id) { return parseFloat(document.getElementById(id).value) || 0; }
  function s(id) { return document.getElementById(id).value; }

  function setLoading(prefix, on) {
    const btn = document.getElementById(prefix+'_btn');
    const bar = document.getElementById(prefix+'_bar');
    btn.disabled = on;
    btn.textContent = on ? 'Calculating...' : (prefix==='c' ? 'Run Cafeteria Forecast' : 'Run Kiosk Forecast');
    bar.style.width = on ? '70%' : '100%';
    if (!on) setTimeout(() => bar.style.width='0%', 600);
  }

  async function predictCafe() {
    document.getElementById('c_error').classList.remove('visible');
    document.getElementById('c_result').classList.remove('visible');
    setLoading('c', true);
    const body = {
      prediction_date: s('c_date'), lag_1: v('c_lag1'), lag_2: v('c_lag2'), lag_3: v('c_lag3'),
      lag_7: v('c_lag7'), lag_14: v('c_lag14'), lag_21: v('c_lag21'),
      roll_mean_7: v('c_rm7'), roll_mean_14: v('c_rm14'), roll_mean_21: v('c_rm21'),
      roll_std_7: v('c_rs7'), roll_std_14: v('c_rs14'), roll_std_21: v('c_rs21'),
      roll_max_7: v('c_rmax7'), roll_max_14: v('c_rmax14'), roll_max_21: v('c_rmax21'),
      roll_min_7: v('c_rmin7'), roll_min_14: v('c_rmin14'), roll_min_21: v('c_rmin21'),
      portions_sold: v('c_psold'), portions_prepared: v('c_pprep'),
      waste_portions: 0, waste_pct: 0,
      ingredient_cost_ugx: v('c_cost'),
      is_weekend: parseInt(s('c_weekend')),
      academic_period: s('c_period'),
      profit_margin: v('c_margin'), waste_ratio: 0.05
    };
    try {
      const r = await fetch('/predict/cafeteria', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || 'Prediction failed');
      document.getElementById('c_amount').textContent = d.predicted_revenue_M;
      document.getElementById('c_sub').textContent = `UGX ${d.predicted_revenue.toLocaleString()}`;
      document.getElementById('c_rdate').textContent = d.prediction_date;
      document.getElementById('c_result').classList.add('visible');
    } catch(e) {
      document.getElementById('c_error').textContent = 'Error: ' + e.message;
      document.getElementById('c_error').classList.add('visible');
    } finally { setLoading('c', false); }
  }

  async function predictKiosk() {
    document.getElementById('k_error').classList.remove('visible');
    document.getElementById('k_result').classList.remove('visible');
    setLoading('k', true);
    const body = {
      prediction_date: s('k_date'), lag_1: v('k_lag1'), lag_2: v('k_lag2'), lag_3: v('k_lag3'),
      lag_7: v('k_lag7'), lag_14: v('k_lag14'), lag_21: v('k_lag21'),
      roll_mean_7: v('k_rm7'), roll_mean_14: v('k_rm14'), roll_mean_21: v('k_rm21'),
      roll_std_7: v('k_rs7'), roll_std_14: v('k_rs14'), roll_std_21: v('k_rs21'),
      roll_max_7: v('k_rmax7'), roll_max_14: v('k_rmax14'), roll_max_21: v('k_rmax21'),
      roll_min_7: v('k_rmin7'), roll_min_14: v('k_rmin14'), roll_min_21: v('k_rmin21'),
      num_transactions: v('k_txn'), avg_unit_price: v('k_price'),
      total_quantity: v('k_qty'), is_weekend: parseInt(s('k_weekend')),
      near_lecture_start: v('k_lecture'), academic_period: s('k_period'),
      avg_transaction_value: v('k_price')
    };
    try {
      const r = await fetch('/predict/kiosk', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || 'Prediction failed');
      document.getElementById('k_amount').textContent = d.predicted_revenue_M;
      document.getElementById('k_sub').textContent = `UGX ${d.predicted_revenue.toLocaleString()}`;
      document.getElementById('k_rdate').textContent = d.prediction_date;
      document.getElementById('k_result').classList.add('visible');
    } catch(e) {
      document.getElementById('k_error').textContent = 'Error: ' + e.message;
      document.getElementById('k_error').classList.add('visible');
    } finally { setLoading('k', false); }
  }
</script>
</body>
</html>""")