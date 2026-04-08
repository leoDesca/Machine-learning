from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "Daily_Prepared",
    "Daily_Sold",
    "Daily_Waste",
    "Daily_Revenue",
    "Daily_Profit",
    "Daily_Sellout_Rate",
    "Daily_Waste_Rate",
    "Daily_Profit_Margin",
    "DOW_sin",
    "DOW_cos",
    "Month_sin",
    "Month_cos",
    "Is_Weekend",
]

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "makerere_Cafeteria_synthetic.csv"


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if den in (0, None):
        return default
    return float(num) / float(den)


def _build_default_profile() -> dict[str, float]:
    """
    Build realistic fallback values for all model features from the
    training dataset. This prevents inference vectors from being dominated
    by zeros when only a minimal set of fields is provided by the UI.
    """
    defaults = {k: 0.0 for k in FEATURE_NAMES}
    if not DATA_FILE.exists():
        return defaults

    cafe = pd.read_csv(DATA_FILE)
    cafe["Date"] = pd.to_datetime(cafe["Date"], errors="coerce")

    for col in ["Price_UGX", "Revenue_UGX", "Ingredient_Cost_UGX", "Waste_Cost_UGX", "Gross_Profit_UGX"]:
        cafe[col] = pd.to_numeric(cafe[col].astype(str).str.replace(",", "", regex=False), errors="coerce")

    cafe["Is_Weekend"] = (
        cafe["Is_Weekend"].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0).astype(int)
    )

    cafe_daily = cafe.groupby(["Date", "Cafeteria_ID"], observed=True).agg(
        Daily_Prepared=("Portions_Prepared", "sum"),
        Daily_Sold=("Portions_Sold", "sum"),
        Daily_Waste=("Waste_Portions", "sum"),
        Daily_Revenue=("Revenue_UGX", "sum"),
        Daily_Ingredient_Cost=("Ingredient_Cost_UGX", "sum"),
        Daily_Waste_Cost=("Waste_Cost_UGX", "sum"),
        Daily_Profit=("Gross_Profit_UGX", "sum"),
        Avg_Waste_Pct=("Waste_Pct", "mean"),
        Waste_Pct_Std=("Waste_Pct", "std"),
    ).reset_index()

    cafe_daily["Daily_Sellout_Rate"] = cafe_daily["Daily_Sold"] / cafe_daily["Daily_Prepared"].replace(0, np.nan)
    cafe_daily["Daily_Waste_Rate"] = cafe_daily["Daily_Waste"] / cafe_daily["Daily_Prepared"].replace(0, np.nan)
    cafe_daily["Daily_Profit_Margin"] = cafe_daily["Daily_Profit"] / cafe_daily["Daily_Revenue"].replace(0, np.nan)
    cafe_daily["Rev_per_Prepared"] = cafe_daily["Daily_Revenue"] / cafe_daily["Daily_Prepared"].replace(0, np.nan)
    cafe_daily["Ingredient_Efficiency"] = cafe_daily["Daily_Profit"] / cafe_daily["Daily_Ingredient_Cost"].replace(0, np.nan)
    cafe_daily["Waste_Cost_Share"] = cafe_daily["Daily_Waste_Cost"] / cafe_daily["Daily_Revenue"].replace(0, np.nan)

    ctx = cafe[["Date", "Cafeteria_ID", "Academic_Period", "Is_Weekend"]].drop_duplicates()
    cafe_daily = cafe_daily.merge(ctx, on=["Date", "Cafeteria_ID"], how="left")

    sold_piv = cafe.pivot_table(
        index=["Date", "Cafeteria_ID"], columns="Meal", values="Portions_Sold", aggfunc="sum"
    ).reset_index().fillna(0)
    sold_piv.columns.name = None
    sold_piv.rename(
        columns={
            "Posho & Beans": "Sold_Posho_Beans",
            "Matooke & Groundnut Stew": "Sold_Matooke_Stew",
            "Rice & Chicken": "Sold_Rice_Chicken",
            "Katogo (Offal+Matoke)": "Sold_Katogo",
            "Chips & Eggs": "Sold_Chips_Eggs",
            "Rolex": "Sold_Rolex",
        },
        inplace=True,
    )
    cafe_daily = cafe_daily.merge(sold_piv, on=["Date", "Cafeteria_ID"], how="left")

    waste_piv = cafe.pivot_table(
        index=["Date", "Cafeteria_ID"], columns="Meal", values="Waste_Pct", aggfunc="mean"
    ).reset_index().fillna(0)
    waste_piv.columns.name = None
    waste_piv.rename(
        columns={
            "Posho & Beans": "WastePct_Posho_Beans",
            "Matooke & Groundnut Stew": "WastePct_Matooke_Stew",
            "Rice & Chicken": "WastePct_Rice_Chicken",
            "Katogo (Offal+Matoke)": "WastePct_Katogo",
            "Chips & Eggs": "WastePct_Chips_Eggs",
            "Rolex": "WastePct_Rolex",
        },
        inplace=True,
    )
    cafe_daily = cafe_daily.merge(waste_piv, on=["Date", "Cafeteria_ID"], how="left")

    sold_cols = [
        "Sold_Posho_Beans",
        "Sold_Matooke_Stew",
        "Sold_Rice_Chicken",
        "Sold_Katogo",
        "Sold_Chips_Eggs",
        "Sold_Rolex",
    ]
    row_tot = cafe_daily[sold_cols].sum(axis=1).replace(0, np.nan)
    shares = cafe_daily[sold_cols].div(row_tot, axis=0).fillna(0)
    cafe_daily["Meal_Entropy"] = -(shares * np.log(shares.replace(0, np.nan))).sum(axis=1).fillna(0)
    cafe_daily["Top_Meal_Share"] = cafe_daily[sold_cols].max(axis=1) / row_tot.fillna(1)

    cafe_daily["Month"] = cafe_daily["Date"].dt.month
    cafe_daily["DayOfWeekNum"] = cafe_daily["Date"].dt.dayofweek
    cafe_daily["Month_sin"] = np.sin(2 * np.pi * (cafe_daily["Month"] - 1) / 12)
    cafe_daily["Month_cos"] = np.cos(2 * np.pi * (cafe_daily["Month"] - 1) / 12)
    cafe_daily["DOW_sin"] = np.sin(2 * np.pi * cafe_daily["DayOfWeekNum"] / 7)
    cafe_daily["DOW_cos"] = np.cos(2 * np.pi * cafe_daily["DayOfWeekNum"] / 7)

    period_map = {
        "Sem1_Teaching": 1,
        "Sem1_Exams": 2,
        "Sem_Break": 3,
        "Sem2_Teaching": 4,
        "Sem2_Exams": 5,
        "Xmas_Break": 6,
        "Other": 0,
    }
    cafe_daily["Period_Code"] = cafe_daily["Academic_Period"].map(period_map).fillna(0).astype(int)
    cafe_daily["Is_Exam_Period"] = cafe_daily["Academic_Period"].isin(["Sem1_Exams", "Sem2_Exams"]).astype(int)
    cafe_daily["Is_Break"] = cafe_daily["Academic_Period"].isin(["Sem_Break", "Xmas_Break"]).astype(int)

    cafe_daily = cafe_daily.sort_values(["Cafeteria_ID", "Date"])
    grp = cafe_daily.groupby("Cafeteria_ID", observed=True)
    cafe_daily["Sold_Roll7_Mean"] = grp["Daily_Sold"].transform(lambda s: s.rolling(7, min_periods=2).mean())
    cafe_daily["Sold_Roll7_Std"] = grp["Daily_Sold"].transform(lambda s: s.rolling(7, min_periods=2).std())
    cafe_daily["Waste_Roll7_Mean"] = grp["Daily_Waste_Rate"].transform(lambda s: s.rolling(7, min_periods=2).mean())
    cafe_daily["Profit_Roll7_Mean"] = grp["Daily_Profit_Margin"].transform(lambda s: s.rolling(7, min_periods=2).mean())
    cafe_daily["Revenue_Roll7_Mean"] = grp["Daily_Revenue"].transform(lambda s: s.rolling(7, min_periods=2).mean())
    cafe_daily["Demand_Z"] = grp["Daily_Sold"].transform(lambda s: (s - s.mean()) / s.std())
    cafe_daily["Revenue_vs_Roll7"] = (
        (cafe_daily["Daily_Revenue"] - cafe_daily["Revenue_Roll7_Mean"])
        / cafe_daily["Revenue_Roll7_Mean"].replace(0, np.nan)
    )

    for col in FEATURE_NAMES:
        if col in cafe_daily.columns:
            defaults[col] = float(cafe_daily[col].replace([np.inf, -np.inf], np.nan).median(skipna=True) or 0.0)

    defaults["Waste_Pct_Std"] = max(defaults.get("Waste_Pct_Std", 0.0), 1.0)
    defaults["Sold_Roll7_Std"] = max(defaults.get("Sold_Roll7_Std", 0.0), 1.0)
    return defaults


DEFAULT_PROFILE = _build_default_profile()


def preprocess_row(cafe_day: dict) -> list[float]:
    """
    Given a dictionary of cafeteria-day data, compute engineered features
    and return the 13-feature vector in the correct order.
    
    Expected input keys:
      daily_prepared, daily_sold, daily_waste, daily_revenue, daily_profit,
      dow (0-6), month (1-12)
    """
    row = pd.Series(DEFAULT_PROFILE).copy()
    row.update(pd.Series(cafe_day))

    prepared = float(row.get("Daily_Prepared", 0) or 0)
    sold = float(row.get("Daily_Sold", 0) or 0)

    # If waste is not supplied by the UI, infer it from prepared and sold.
    # This keeps waste-related features coherent with minimal-form inputs.
    if "Daily_Waste" not in cafe_day:
        row["Daily_Waste"] = max(prepared - sold, 0.0)

    # Core ratios from totals
    row["Daily_Sellout_Rate"] = _safe_div(row.get("Daily_Sold", 0), row.get("Daily_Prepared", 0), row["Daily_Sellout_Rate"])
    row["Daily_Waste_Rate"] = _safe_div(row.get("Daily_Waste", 0), row.get("Daily_Prepared", 0), row["Daily_Waste_Rate"])
    row["Daily_Profit_Margin"] = _safe_div(row.get("Daily_Profit", 0), row.get("Daily_Revenue", 0), row["Daily_Profit_Margin"])

    # Time-based cyclic features
    month = int(row.get("Month", 1))

    # UI uses 0=Sun..6=Sat. Training used pandas dayofweek 0=Mon..6=Sun.
    # Convert UI convention to training convention for consistency.
    ui_dow = int(row.get("DayOfWeekNum", 0))
    dow = (ui_dow - 1) % 7
    
    row["Month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    row["Month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    row["DOW_sin"] = np.sin(2 * np.pi * dow / 7)
    row["DOW_cos"] = np.cos(2 * np.pi * dow / 7)

    # Weekend in UI convention: Sunday (0) and Saturday (6).
    row["Is_Weekend"] = 1 if ui_dow in (0, 6) else 0

    return [float(row.get(f, DEFAULT_PROFILE.get(f, 0))) for f in FEATURE_NAMES]
