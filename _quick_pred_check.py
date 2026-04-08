import json
import joblib
import hdbscan
import numpy as np
from preprocess import preprocess_row

clusterer = joblib.load('model/hdbscan_model.pkl')
scaler = joblib.load('model/scaler.pkl')

def pred(payload):
    v = np.array(preprocess_row(payload), dtype=float).reshape(1,-1)
    x = scaler.transform(v)
    labels, strengths = hdbscan.approximate_predict(clusterer, x)
    return int(labels[0]), float(strengths[0])

samples = {
    'normal-ish': {'Daily_Prepared': 600, 'Daily_Sold': 520, 'Daily_Revenue': 2200000, 'Daily_Profit': 780000, 'DayOfWeekNum': 2, 'Month': 4},
    'high-volume': {'Daily_Prepared': 2400, 'Daily_Sold': 2100, 'Daily_Revenue': 9600000, 'Daily_Profit': 3600000, 'DayOfWeekNum': 1, 'Month': 9},
    'low-day': {'Daily_Prepared': 90, 'Daily_Sold': 40, 'Daily_Revenue': 180000, 'Daily_Profit': 30000, 'DayOfWeekNum': 6, 'Month': 1}
}

for name,p in samples.items():
    c,s = pred(p)
    print(name, '=> cluster', c, 'strength', round(s,4))
