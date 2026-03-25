"""
test_api.py — API endpoint tests
----------------------------------
Run after the server is up:
    python test_api.py

Uses only Python's built-in urllib — no requests library needed.
"""

import json
import urllib.request
import urllib.error

BASE = "http://localhost:8000"
PASS = FAIL = 0


def test(name, method, path, body=None, expected_status=200):
    global PASS, FAIL
    data = json.dumps(body).encode() if body else None
    req  = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        status = e.code
        result = json.loads(e.read())
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        FAIL += 1
        return

    ok = (status == expected_status)
    print(f"  {'[PASS]' if ok else '[FAIL]'} {name}  →  HTTP {status}")
    if ok:
        PASS += 1
    else:
        FAIL += 1
        return

    if "cluster_name" in result:
        print(f"         Cluster    : {result['cluster']} — {result['cluster_name']}")
        print(f"         Alert      : {result['supply_alert']}")
        print(f"         Strength   : {result['membership_strength']}  ({result['confidence']})")
        print(f"         Is noise   : {result['is_noise']}")
    if "status" in result:
        print(f"         Status     : {result['status']}")


# Standard Africa Hall day (high volume → Cluster 1)
AFRICA_HALL_DAY = {
    "daily_prepared": 2200, "daily_sold": 2067, "daily_waste": 133,
    "daily_revenue": 8268000, "daily_profit": 3180000,
    "daily_sellout_rate": 0.94, "daily_waste_rate": 0.06,
    "daily_profit_margin": 0.38, "rev_per_prepared": 3758,
    "ingredient_efficiency": 1.42, "waste_cost_share": 0.07,
    "avg_waste_pct": 6.0, "waste_pct_std": 1.2,
    "sold_posho_beans": 415, "sold_matooke_stew": 330, "sold_rice_chicken": 280,
    "sold_katogo": 210, "sold_chips_eggs": 250, "sold_rolex": 582,
    "meal_entropy": 1.76, "top_meal_share": 0.28,
    "wastepct_posho_beans": 6.1, "wastepct_matooke_stew": 5.8,
    "wastepct_rice_chicken": 6.4, "wastepct_katogo": 6.2,
    "wastepct_chips_eggs": 5.9, "wastepct_rolex": 5.7,
    "dow": 1, "month": 3,
    "is_weekend": 0, "period_code": 1, "is_exam_period": 0, "is_break": 0,
    "sold_roll7_mean": 2010, "sold_roll7_std": 120,
    "waste_roll7_mean": 0.061, "profit_roll7_mean": 0.375,
    "demand_z": 0.8, "revenue_vs_roll7": 0.05,
}

# Small cafeteria, semester break (noise candidate)
BREAK_DAY = {
    "daily_prepared": 80, "daily_sold": 52, "daily_waste": 28,
    "daily_revenue": 156000, "daily_profit": 42000,
    "daily_sellout_rate": 0.65, "daily_waste_rate": 0.35,
    "daily_profit_margin": 0.27, "rev_per_prepared": 1950,
    "ingredient_efficiency": 0.62, "waste_cost_share": 0.22,
    "avg_waste_pct": 34.5, "waste_pct_std": 8.1,
    "sold_posho_beans": 20, "sold_matooke_stew": 12, "sold_rice_chicken": 6,
    "sold_katogo": 4, "sold_chips_eggs": 5, "sold_rolex": 5,
    "meal_entropy": 1.62, "top_meal_share": 0.38,
    "wastepct_posho_beans": 33.0, "wastepct_matooke_stew": 35.0,
    "wastepct_rice_chicken": 38.0, "wastepct_katogo": 34.0,
    "wastepct_chips_eggs": 32.0, "wastepct_rolex": 31.0,
    "dow": 3, "month": 6,
    "is_weekend": 0, "period_code": 3, "is_exam_period": 0, "is_break": 1,
    "sold_roll7_mean": 58, "sold_roll7_std": 22,
    "waste_roll7_mean": 0.32, "profit_roll7_mean": 0.28,
    "demand_z": -2.1, "revenue_vs_roll7": -0.15,
}

# Normal cafeteria, teaching day (Cluster 0)
NORMAL_DAY = {
    "daily_prepared": 450, "daily_sold": 393, "daily_waste": 57,
    "daily_revenue": 1572000, "daily_profit": 609000,
    "daily_sellout_rate": 0.873, "daily_waste_rate": 0.127,
    "daily_profit_margin": 0.387, "rev_per_prepared": 3493,
    "ingredient_efficiency": 1.28, "waste_cost_share": 0.11,
    "avg_waste_pct": 12.8, "waste_pct_std": 3.4,
    "sold_posho_beans": 88, "sold_matooke_stew": 70, "sold_rice_chicken": 55,
    "sold_katogo": 42, "sold_chips_eggs": 50, "sold_rolex": 88,
    "meal_entropy": 1.77, "top_meal_share": 0.22,
    "wastepct_posho_beans": 12.8, "wastepct_matooke_stew": 12.5,
    "wastepct_rice_chicken": 13.2, "wastepct_katogo": 13.8,
    "wastepct_chips_eggs": 12.9, "wastepct_rolex": 12.5,
    "dow": 0, "month": 9,
    "is_weekend": 0, "period_code": 4, "is_exam_period": 0, "is_break": 0,
    "sold_roll7_mean": 395, "sold_roll7_std": 28,
    "waste_roll7_mean": 0.128, "profit_roll7_mean": 0.384,
    "demand_z": 0.1, "revenue_vs_roll7": 0.02,
}

print("=" * 58)
print("  Group 17 — HDBSCAN API endpoint tests")
print("=" * 58)

test("GET /health", "GET", "/health")
test("GET /clusters", "GET", "/clusters")
test("GET /api/dashboard", "GET", "/api/dashboard")
test("GET /api/inventory/alerts", "GET", "/api/inventory/alerts")
test("POST /predict — Africa Hall day (expect Cluster 1)", "POST", "/predict", body=AFRICA_HALL_DAY)
test("POST /predict — Normal cafeteria day (expect Cluster 0)", "POST", "/predict", body=NORMAL_DAY)
test("POST /predict — Semester break day (may be noise)", "POST", "/predict", body=BREAK_DAY)
test("POST /predict — Missing fields (expect 400)", "POST", "/predict",
     body={"daily_sold": 400}, expected_status=400)
test("GET /unknown (expect 404)", "GET", "/unknown", expected_status=404)

print("=" * 58)
print(f"  Results: {PASS} passed, {FAIL} failed")
print("=" * 58)
