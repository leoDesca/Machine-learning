import random
import numpy as np
import joblib
import hdbscan
from preprocess import preprocess_row

clusterer = joblib.load('model/hdbscan_model.pkl')
scaler = joblib.load('model/scaler.pkl')

found = []
for _ in range(12000):
    payload = {
        'Daily_Prepared': random.uniform(80, 3000),
        'Daily_Sold': random.uniform(50, 2600),
        'Daily_Revenue': random.uniform(200000, 12000000),
        'Daily_Profit': random.uniform(50000, 4500000),
        'DayOfWeekNum': random.randint(0,6),
        'Month': random.randint(1,12),
    }
    x = np.array(preprocess_row(payload), dtype=float).reshape(1,-1)
    xs = scaler.transform(x)
    labels, strengths = hdbscan.approximate_predict(clusterer, xs)
    c = int(labels[0])
    if c == 0:
        found.append((float(strengths[0]), payload))
        if len(found) >= 3:
            break

print('cluster0_found_count=', len(found))
for s,p in found:
    print(round(s,4), p)
