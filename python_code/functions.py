import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as sm
from pykalman import KalmanFilter
import statsmodels.api as smf
import yfinance as yf

class DatasetGenerator:
    def __init__(self, symbols):
        self.symbols = symbols

    def get_symbols(self, data_source='yahoo', ohlc='Close', start_date=None, end_date=None):
        out = []
        new_symbols = []
        for symbol in self.symbols:
            df = yf.download(symbol, start=start_date, end=end_date)
            df = df[ohlc]
            new_symbols.append(symbol)
            out.append(df.astype('float'))
        data = pd.concat(out, axis=1)
        data.columns = new_symbols
        data = data.dropna(axis=1)
        return data.dropna(axis=1)

    def train_test_split(self, data, train_size=0.8):
        train_size = int(len(data) * train_size)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        return train_data, test_data

    def get_train_test_data(self, data_source='yahoo', ohlc='Close', start_date=None, end_date=None, train_size=0.8):
        prices = self.get_symbols(data_source, ohlc, start_date, end_date)
        train_data, test_data = self.train_test_split(prices, train_size)
        return train_data, test_data

class Functions:
    def find_cointegrated_pairs(self, dataframe, critical_level=0.05):
        n = dataframe.shape[1]
        pvalue_matrix = np.ones((n, n))  # initialize the matrix (nxn) of p_value with 1
        keys = dataframe.columns  # get the column names
        pairs = []  # initialize the list for cointegration
        for i in range(n):
            for j in range(i+1, n):  # for j bigger than i
                stock1 = dataframe[keys[i]]  # obtain the price of "stock1"
                stock2 = dataframe[keys[j]]  # obtain the price of "stock2"
                result = sm.coint(stock1, stock2)  # get cointegration
                pvalue = result[1]  # get the pvalue
                pvalue_matrix[i, j] = pvalue
                if pvalue < critical_level:  # if p-value less than the critical level
                    pairs.append((keys[i], keys[j], pvalue))  # record the contract with that p-value
        pairs = pd.DataFrame(pairs)
        pairs.columns = ["Asset.1", "Asset.2", "p_value"]
        return pvalue_matrix, pairs

    def KalmanFilterAverage(self, x):
        # Construct a Kalman filter
        kf = KalmanFilter(transition_matrices=[1],
                          observation_matrices=[1],
                          initial_state_mean=0,
                          initial_state_covariance=1,
                          observation_covariance=1,
                          transition_covariance=.01)
        # Use the observed values of the price to get a rolling mean
        state_means, _ = kf.filter(x.values)
        # Aligning the results with the original time series index
        state_means = pd.Series(state_means.flatten(), index=x.index)
        return state_means

    def KalmanFilterRegression(self, x, y):
        delta = 1e-3
        trans_cov = delta / (1 - delta) * np.eye(2)  # How much random walk wiggles
        obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
        kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,  # y is 1-dimensional, (alpha, beta) is 2-dimensional
                          initial_state_mean=[0, 0],
                          initial_state_covariance=np.ones((2, 2)),
                          transition_matrices=np.eye(2),
                          observation_matrices=obs_mat,
                          observation_covariance=2,
                          transition_covariance=trans_cov)
        # Use the observations y to get running estimates and errors for the state parameters
        state_means, state_covs = kf.filter(y.values)
        return state_means

    def half_life(self, spread):
        spread_lag = spread.shift(1)
        spread_lag.iloc[0] = spread_lag.iloc[1]
        spread_ret = spread - spread_lag
        spread_ret.iloc[0] = spread_ret.iloc[1]
        spread_lag2 = smf.add_constant(spread_lag)
        model = smf.OLS(spread_ret, spread_lag2)
        res = model.fit()
        halflife = int(round(-np.log(2) / res.params[1], 0))
        if halflife <= 0:
            halflife = 1
        return halflife


