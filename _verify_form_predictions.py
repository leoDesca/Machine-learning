from app import _predict_from_payload

# UI convention: 0=Sun..6=Sat
samples = {
    'normal-ish': {'daily_prepared': 600, 'daily_sold': 520, 'daily_revenue': 2200000, 'daily_profit': 780000, 'dow': 2, 'month': 4},
    'high-volume candidate A': {'daily_prepared': 2150, 'daily_sold': 2000, 'daily_revenue': 7600000, 'daily_profit': 2980000, 'dow': 2, 'month': 10},
    'high-volume candidate B': {'daily_prepared': 2300, 'daily_sold': 2050, 'daily_revenue': 7800000, 'daily_profit': 3050000, 'dow': 3, 'month': 11},
    'low-day': {'daily_prepared': 90, 'daily_sold': 40, 'daily_revenue': 180000, 'daily_profit': 30000, 'dow': 6, 'month': 1}
}

for name, payload in samples.items():
    out = _predict_from_payload(payload)
    print(name, '=>', out['cluster'], '|', out['cluster_name'], '| strength', out['membership_strength'])
