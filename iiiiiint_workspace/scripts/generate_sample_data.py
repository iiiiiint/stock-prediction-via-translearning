import os
import numpy as np
import pandas as pd

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
raw_dir = os.path.join(root, 'data', 'raw')
os.makedirs(raw_dir, exist_ok=True)

# generate 300 business days
dates = pd.bdate_range(end=pd.Timestamp.today(), periods=600)

def make_df(dates, seed=0, start_price=100.0):
    np.random.seed(seed)
    n = len(dates)
    # random returns ~ N(0, 0.01)
    rets = np.random.normal(loc=0.0002, scale=0.02, size=n)
    price = start_price * np.exp(np.cumsum(rets))
    open_p = price * (1 + np.random.normal(0, 0.002, size=n))
    high = np.maximum(open_p, price) * (1 + np.abs(np.random.normal(0, 0.01, size=n)))
    low = np.minimum(open_p, price) * (1 - np.abs(np.random.normal(0, 0.01, size=n)))
    close = price
    volume = np.random.randint(1000, 10000, size=n)
    df = pd.DataFrame({
        'date': dates,
        'open': open_p,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })
    return df

# target example (CN_300750)
df_target = make_df(dates, seed=1, start_price=60.0)
df_target.to_csv(os.path.join(raw_dir, 'CN_300750.csv'), index=False)

# US_NDX example
df_ndx = make_df(dates, seed=2, start_price=15000.0)
df_ndx.to_csv(os.path.join(raw_dir, 'US_NDX.csv'), index=False)

print('Sample data written to', raw_dir)
