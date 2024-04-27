#%%
import backtrader as bt
import backtrader.feeds as btfeeds
import yfinance as yf
import datetime as dt

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from pykalman import KalmanFilter

import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
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

# %%
ticker_list = list(price_df.columns)
num_stocks = price_df.shape[1]
print('Number of Stocks =', num_stocks) 

# %%
def KalmanFilterAverage(x):
  # Construct a Kalman filter
    kf = KalmanFilter(transition_matrices = [1],
    observation_matrices = [1],
    initial_state_mean = 0,
    initial_state_covariance = 1,
    observation_covariance=1,
    transition_covariance=.01)
  # Use the observed values of the price to get a rolling mean
    state_means, _ = kf.filter(x.values)
  # Aligning the results with the original time series index
    state_means = pd.Series(state_means.flatten(), index=x.index)
    return state_means

# Kalman filter regression
def KalmanFilterRegression(x,y):
    delta = 1e-3
    trans_cov = delta / (1 - delta) * np.eye(2) # How much random walk wiggles
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, # y is 1-dimensional, (alpha, beta) is 2-dimensional
    initial_state_mean=[0,0],
    initial_state_covariance=np.ones((2, 2)),
    transition_matrices=np.eye(2),
    observation_matrices=obs_mat,
    observation_covariance=2,
    transition_covariance=trans_cov)
    # Use the observations y to get running estimates and errors for the state parameters
    state_means, state_covs = kf.filter(y.values)
    return state_means

def half_life(spread):
    spread_lag = spread.shift(1)
    spread_lag.iloc[0] = spread_lag.iloc[1]
    spread_ret = spread - spread_lag
    spread_ret.iloc[0] = spread_ret.iloc[1]
    spread_lag2 = sm.add_constant(spread_lag)
    model = sm.OLS(spread_ret,spread_lag2)
    res = model.fit()
    # print(res.params.iloc[1])
    halflife = int(round(-np.log(2) / res.params.iloc[1], 0))
    if halflife <= 0:
        halflife = 1
    return halflife

# %%
training_split = int(df.shape[0] * 0.8)
print(f"We use data up to {str(df.index[training_split-1])[:10]} to find cointegration pairs.")
print()

cointegration_df = pd.DataFrame(index = ticker_list, columns = ticker_list)
cointegration_pairs = []

for i, ticker_1 in enumerate(ticker_list):
    for j, ticker_2 in enumerate(ticker_list):
        if i == j:
            continue
        p_value = coint(price_df.iloc[:training_split, i], price_df.iloc[:training_split, j])[1]
        cointegration_df.loc[ticker_1, ticker_2] = p_value
        if i > j and p_value < 0.05:
            print("Good cointegration pairs:", ticker_1, ticker_2, p_value)
            cointegration_pairs.append([ticker_1, ticker_2])

print()
print('Number of Pairs =', len(cointegration_pairs)) 

# %%
# Heatmap
sns.heatmap(cointegration_df.astype(float), xticklabels=ticker_list,yticklabels=ticker_list)
plt.title('Heatmap of Cointegration P-Values')
plt.xticks(ha='right', size=8)
plt.yticks(size=8)
plt.tight_layout()
plt.show();

#%%
column_name = [f"Pair {str(i+1)}" for i in range(len(cointegration_pairs))]
train_df = price_df.iloc[:training_split, :]
test_df = price_df.iloc[training_split:, :]

pair_df = pd.DataFrame(index = column_name, columns = ["Ticker 1", "Ticker 2"])
train_pair_spread = pd.DataFrame(columns = column_name)
train_half_life = pd.DataFrame(index = column_name, columns = ["Half Life"])
test_pair_spread = pd.DataFrame(columns = column_name)
test_half_life = pd.DataFrame(index = column_name, columns = ["Half Life"])

for i in range(len(cointegration_pairs)):
    pair_df.iloc[i, :] = cointegration_pairs[i]
    
    x = train_df[cointegration_pairs[i][0]]
    y = train_df[cointegration_pairs[i][1]]
    state_means = KalmanFilterRegression(KalmanFilterAverage(x), KalmanFilterAverage(y))
    hedge_ratio = state_means[:, 0]
    spread = y - x * hedge_ratio
    train_pair_spread.iloc[:, i] = spread
    train_half_life.iloc[i, 0] = half_life(spread)
    
    x = test_df[cointegration_pairs[i][0]]
    y = test_df[cointegration_pairs[i][1]]
    state_means = KalmanFilterRegression(KalmanFilterAverage(x), KalmanFilterAverage(y))
    hedge_ratio = state_means[:, 0]
    spread = y - x * hedge_ratio
    test_pair_spread.iloc[:, i] = spread
    test_half_life.iloc[i, 0] = half_life(spread)
    
pair_df.to_csv("pair_df.csv")
train_pair_spread.to_csv("train_pair_spread.csv")
train_half_life.to_csv("train_half_life.csv")
test_pair_spread.to_csv("test_pair_spread.csv")
test_half_life.to_csv("test_half_life.csv")

