import pandas as pd
import numpy as np
import joblib
import hdbscan

from preprocess import preprocess_row

clusterer = joblib.load('model/hdbscan_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Use original cafeteria dataset to estimate realistic 6-field ranges per cluster.
df = pd.read_csv('makerere_Cafeteria_synthetic.csv')
df['Date'] = pd.to_datetime(df['Date'])
for c in ['Revenue_UGX','Gross_Profit_UGX']:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace(',','', regex=False), errors='coerce')

# aggregate to daily level (matching API semantics)
d = df.groupby(['Date','Cafeteria_ID'], observed=True).agg(
    daily_prepared=('Portions_Prepared','sum'),
    daily_sold=('Portions_Sold','sum'),
    daily_revenue=('Revenue_UGX','sum'),
    daily_profit=('Gross_Profit_UGX','sum')
).reset_index()

d['dow'] = d['Date'].dt.dayofweek
# app UI says 0=Sun..6=Sat; convert from pandas 0=Mon..6=Sun
# for model input mapping it ultimately just becomes cyclical so keep simple with pandas index mapping alternative
# we'll map to app convention here:
d['dow_app'] = (d['dow'] + 1) % 7

d['month'] = d['Date'].dt.month

rows=[]
for _,r in d.iterrows():
    payload = {
        'Daily_Prepared': float(r['daily_prepared']),
        'Daily_Sold': float(r['daily_sold']),
        'Daily_Revenue': float(r['daily_revenue']),
        'Daily_Profit': float(r['daily_profit']),
        'DayOfWeekNum': int(r['dow']),
        'Month': int(r['month'])
    }
    x = np.array(preprocess_row(payload), dtype=float).reshape(1,-1)
    xs = scaler.transform(x)
    label, strength = hdbscan.approximate_predict(clusterer, xs)
    rows.append((int(label[0]), float(strength[0]), payload['Daily_Prepared'], payload['Daily_Sold'], payload['Daily_Revenue'], payload['Daily_Profit'], int(r['dow_app']), int(r['month'])))

res = pd.DataFrame(rows, columns=['cluster','strength','daily_prepared','daily_sold','daily_revenue','daily_profit','dow','month'])

for c in sorted(res['cluster'].unique()):
    sub = res[res['cluster']==c]
    print(f'\nCluster {c}: n={len(sub)}')
    for col in ['daily_prepared','daily_sold','daily_revenue','daily_profit']:
        q = sub[col].quantile([0.1,0.5,0.9]).to_dict()
        print(f"  {col}: p10={q[0.1]:.1f}, p50={q[0.5]:.1f}, p90={q[0.9]:.1f}")
    print('  dow mode:', sub['dow'].mode().tolist()[:3], 'month mode:', sub['month'].mode().tolist()[:3])

# print a few high-confidence examples for cluster 1 and 0
for target in [1,0,-1]:
    s = res[res['cluster']==target].sort_values('strength', ascending=False).head(5)
    if len(s):
        print(f'\nTop examples cluster {target}:')
        print(s[['strength','daily_prepared','daily_sold','daily_revenue','daily_profit','dow','month']].to_string(index=False))
