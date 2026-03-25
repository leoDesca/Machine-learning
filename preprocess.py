import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "Daily_Prepared","Daily_Sold","Daily_Waste","Daily_Revenue","Daily_Profit",
    "Daily_Sellout_Rate","Daily_Waste_Rate","Daily_Profit_Margin",
    "Rev_per_Prepared","Ingredient_Efficiency","Waste_Cost_Share",
    "Avg_Waste_Pct","Waste_Pct_Std",
    "Sold_Posho_Beans","Sold_Matooke_Stew","Sold_Rice_Chicken",
    "Sold_Katogo","Sold_Chips_Eggs","Sold_Rolex",
    "Meal_Entropy","Top_Meal_Share",
    "WastePct_Posho_Beans","WastePct_Matooke_Stew","WastePct_Rice_Chicken",
    "WastePct_Katogo","WastePct_Chips_Eggs","WastePct_Rolex",
    "DOW_sin","DOW_cos","Month_sin","Month_cos",
    "Period_Code","Is_Weekend","Is_Exam_Period","Is_Break",
    "Sold_Roll7_Mean","Sold_Roll7_Std","Waste_Roll7_Mean",
    "Profit_Roll7_Mean","Demand_Z","Revenue_vs_Roll7",
]

def preprocess_row(cafe_day: dict) -> list[float]:
    """
    Given a dictionary of cafeteria-day data, compute engineered features
    and return the 41-feature vector in the correct order.
    """
    row = pd.Series(cafe_day)

    # Example engineered features (simplified from train.py)
    row["Daily_Sellout_Rate"] = row.get("Daily_Sold",0) / max(row.get("Daily_Prepared",1),1)
    row["Daily_Waste_Rate"]   = row.get("Daily_Waste",0) / max(row.get("Daily_Prepared",1),1)
    row["Daily_Profit_Margin"]= row.get("Daily_Profit",0) / max(row.get("Daily_Revenue",1),1)
    row["Rev_per_Prepared"]   = row.get("Daily_Revenue",0) / max(row.get("Daily_Prepared",1),1)
    row["Ingredient_Efficiency"] = row.get("Daily_Profit",0) / max(row.get("Daily_Ingredient_Cost",1),1)
    row["Waste_Cost_Share"]   = row.get("Daily_Waste_Cost",0) / max(row.get("Daily_Revenue",1),1)

    # Time-based features
    if "Month" in row:
        row["Month_sin"] = np.sin(2*np.pi*(row["Month"]-1)/12)
        row["Month_cos"] = np.cos(2*np.pi*(row["Month"]-1)/12)
    if "DayOfWeekNum" in row:
        row["DOW_sin"] = np.sin(2*np.pi*row["DayOfWeekNum"]/7)
        row["DOW_cos"] = np.cos(2*np.pi*row["DayOfWeekNum"]/7)

    # Fill missing with 0
    return [float(row.get(f,0)) for f in FEATURE_NAMES]
