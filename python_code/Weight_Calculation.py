#%%
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint

#%%
symbols = ['GDX','GDXJ','GLD', 'AAPL','GOOGL', 'META','TWTR','AMD',
           'NVDA','CSCO', 'ORCL', 'ATVI', 'TTWO', 'EA', 'HYG', 
           'LQD', 'JNK', 'SLV', 'USLV', 'SIVR', 'USO', 'UWT', 
           'QQQ', 'SPY', 'VOO', 'VDE', 'VTI', 'EMLP', 'VDC', 
           'FSTA', 'KXI', 'IBB', 'VHT','VNQ', 'IYR', 'MSFT', 
           'PG', 'TMF', 'UPRO', 'WFC', 'JPM', 'GS', 'CVX', 
           'XOM', 'INTC', 'COST', 'WMT', 'T', 'VZ', 'CMCSA', 'AMZN']

start = pd.Timestamp('2012-01-01')
end = pd.Timestamp('2023-12-31')

price = {}
price_df = pd.DataFrame()

for symbol in symbols:
    df = yf.download(symbol, start= start, end = end)
    price[symbol] = df
    price_df[symbol] = df["Close"]

price_df.dropna(axis = 1, inplace = True)

#%%
cointegration_pairs = []
pair_df = pd.read_csv("output/pair_df.csv", header = 0, index_col = 0)
for i in range(pair_df.shape[0]):
    cointegration_pairs.append([pair_df.iloc[i, 0], pair_df.iloc[i, 1]])
    
test_spread_vol = pd.read_csv("output/test_spread_vol.csv", header = 0, index_col = 0)
# %%
look_back_period = 250

rolling_coint_p = pd.DataFrame(index = test_spread_vol.index, columns = test_spread_vol.columns)
rolling_coint_t = pd.DataFrame(index = test_spread_vol.index, columns = test_spread_vol.columns)
rolling_coint_p.index = pd.to_datetime(rolling_coint_p.index)

for i in range(rolling_coint_p.shape[0]):
    today_row = list(price_df.index).index(rolling_coint_p.index[i])
    last = today_row - (look_back_period - 1)
    tdy = today_row + 1
    for j in range(len(cointegration_pairs)):
        result = coint(price_df[cointegration_pairs[j][0]].iloc[last:tdy], price_df[cointegration_pairs[j][1]].iloc[last:tdy])
        rolling_coint_p.iloc[i, j] = result[1]
        rolling_coint_t.iloc[i, j] = abs(result[0])

rolling_coint_p.to_csv("output/rolling_coint_pvalue.csv")
rolling_coint_t.to_csv("output/rolling_coint_tstat.csv")

# %%
confidence_p_df = pd.DataFrame(index = test_spread_vol.index, columns = test_spread_vol.columns)
confidence_t_df = pd.DataFrame(index = test_spread_vol.index, columns = test_spread_vol.columns)

for i in range(rolling_coint_p.shape[0]):
    for j in range(rolling_coint_p.shape[1]):
        confidence_p_df.iloc[i, j] = test_spread_vol.iloc[i, j] / rolling_coint_p.iloc[i, j]
        confidence_t_df.iloc[i, j] = test_spread_vol.iloc[i, j] * rolling_coint_t.iloc[i, j]
        
# %%
weight_p_df = pd.DataFrame(index = test_spread_vol.index, columns = test_spread_vol.columns)
weight_t_df = pd.DataFrame(index = test_spread_vol.index, columns = test_spread_vol.columns)
max_weight_allowed = 1 / len(cointegration_pairs) * 4

def allocate_weight(weight, confidence, row):
    total_weight = 1 - weight.iloc[row, :].sum()
    if total_weight <= 0:
        return weight
    confidence_sum = 0
    for j in range(weight.shape[1]):
        if np.isnan(weight.iloc[row, j]):
            confidence_sum += confidence.iloc[row, j]
    if confidence_sum <= 0:
        return weight
    for j in range(weight.shape[1]):
        if np.isnan(weight.iloc[row, j]):
            allocated_weight = confidence.iloc[row, j] / confidence_sum
            if allocated_weight > total_weight * max_weight_allowed:
                weight.iloc[row, j] = total_weight * max_weight_allowed
    return weight

for i in range(weight_p_df.shape[0]):
    count = 0
    while weight_p_df.iloc[i, :].dropna().shape != weight_p_df.iloc[i, :].shape and count < 200:
        weight_p_df = allocate_weight(weight_p_df, confidence_p_df, i)
        count += 1    
    for j in range(weight_p_df.shape[1]):
        if weight_p_df.iloc[i, j] >= max_weight_allowed:
            weight_p_df.iloc[i, j] = max_weight_allowed
        elif np.isnan(weight_p_df.iloc[i, j]):
            if np.isnan(weight_p_df.iloc[i, :].min()):
                weight_p_df.iloc[i, j] = 1 / len(cointegration_pairs)
            else:
                weight_p_df.iloc[i, j] = weight_p_df.iloc[i, :].min()
    if weight_p_df.iloc[i, :].sum() != 0:
        weight_p_df.iloc[i, :] = weight_p_df.iloc[i, :] / weight_p_df.iloc[i, :].sum()
    
    count = 0
    while weight_t_df.iloc[i, :].dropna().shape != weight_t_df.iloc[i, :].shape and count < 200:
        weight_t_df = allocate_weight(weight_t_df, confidence_t_df, i)
        count += 1
    for j in range(weight_t_df.shape[1]):
        if weight_t_df.iloc[i, j] >= max_weight_allowed:
            weight_t_df.iloc[i, j] = max_weight_allowed
        elif np.isnan(weight_t_df.iloc[i, j]):
            if np.isnan(weight_t_df.iloc[i, :].min()):
                weight_t_df.iloc[i, j] = 1 / len(cointegration_pairs)
            else:
                weight_t_df.iloc[i, j] = weight_t_df.iloc[i, :].min()
    if weight_t_df.iloc[i, :].sum() != 0:
        weight_t_df.iloc[i, :] = weight_t_df.iloc[i, :] / weight_t_df.iloc[i, :].sum()
        
weight_p_df.to_csv("output/weight_p_df.csv")
weight_t_df.to_csv("output/weight_t_df.csv")
# %%
