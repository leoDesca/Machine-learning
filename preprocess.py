FEATURE_NAMES = [
    "Daily_Prepared",
    "Daily_Sold",
    "Daily_Revenue",
    "Daily_Profit",
    "DayOfWeekNum",
    "Month",
]


def preprocess_row(cafe_day: dict) -> list[float]:
    """
    Build the feature vector used by the deployed model.

    This deployment intentionally uses the exact 6 fields exposed by the UI,
    so training and inference stay perfectly aligned.
    """
    row = {
        "Daily_Prepared": float(cafe_day.get("Daily_Prepared", 0.0)),
        "Daily_Sold": float(cafe_day.get("Daily_Sold", 0.0)),
        "Daily_Revenue": float(cafe_day.get("Daily_Revenue", 0.0)),
        "Daily_Profit": float(cafe_day.get("Daily_Profit", 0.0)),
        "DayOfWeekNum": float(cafe_day.get("DayOfWeekNum", 0.0)),
        "Month": float(cafe_day.get("Month", 0.0)),
    }

    return [row[f] for f in FEATURE_NAMES]
