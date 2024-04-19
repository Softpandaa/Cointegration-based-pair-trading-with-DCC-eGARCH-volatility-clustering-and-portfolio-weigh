import pandas as pd
import matplotlib.pyplot as plt
from functions import DatasetGenerator
from functions import Functions

symbols = ['GDX', 'GDXJ', 'GLD', 'AAPL', 'GOOGL', 'META', 'TWTR', 'AMD',
               'NVDA', 'CSCO', 'ORCL', 'ATVI', 'TTWO', 'EA', 'HYG',
               'LQD', 'JNK', 'SLV', 'USLV', 'SIVR', 'USO', 'UWT',
               'QQQ', 'SPY', 'VOO', 'VDE', 'VTI', 'EMLP', 'VDC',
               'FSTA', 'KXI', 'IBB', 'VHT', 'VNQ', 'IYR', 'MSFT',
               'PG', 'TMF', 'UPRO', 'WFC', 'JPM', 'GS', 'CVX',
               'XOM', 'INTC', 'COST', 'WMT', 'T', 'VZ', 'CMCSA', 'AMZN']

start_date = '2012-01-01'
end_date = '2023-12-31'

dataset_generator = DatasetGenerator(symbols)
train_data, test_data = dataset_generator.get_train_test_data(data_source='yahoo', ohlc='Close', start_date=start_date, end_date=end_date, train_size=0.8)

print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

functions = Functions()
pvalue_matrix, cointegrated_pairs = functions.find_cointegrated_pairs(train_data)
cointegrated_pairs

# Graphs -----------
'''
for pair in range(len(cointegrated_pairs)):
    pair1_name = cointegrated_pairs.iloc[pair,0]
    pair2_name = cointegrated_pairs.iloc[pair,1]
    p_value = cointegrated_pairs.iloc[pair,2]
    # print(pair1_name, pair2_name, p_value)
    plt.plot(train_data[pair1_name], label=pair1_name)
    plt.plot(train_data[pair2_name], label=pair2_name)
    plt.title(f'Rolling alpha & Beta for {pair1_name} and {pair2_name}')
    plt.legend()
    plt.show()
'''


train_spread = {}
for pair in range(len(cointegrated_pairs)):
    pair1_name = cointegrated_pairs.iloc[pair,0]
    pair2_name = cointegrated_pairs.iloc[pair,1]
    x = train_data[pair1_name]
    y = train_data[pair2_name]
    df1 = pd.DataFrame({'y': y, 'x': x})
    df1.index = pd.to_datetime(df1.index)
    state_means = functions.KalmanFilterRegression(functions.KalmanFilterAverage(x), functions.KalmanFilterAverage(y))
    df1['hr'] = abs(state_means[:, 0])  # Hedge Ratio
    df1['spread'] = df1.y - (df1.x * df1.hr)
    train_spread[f"{pair1_name}_{pair2_name}"] = df1['spread']
train_spread = pd.DataFrame(train_spread)
train_spread.index = pd.to_datetime(train_data.index)

test_spread = {}
for pair in range(len(cointegrated_pairs)):
    pair1_name = cointegrated_pairs.iloc[pair,0]
    pair2_name = cointegrated_pairs.iloc[pair,1]
    x = test_data[pair1_name]
    y = test_data[pair2_name]
    df1 = pd.DataFrame({'y': y, 'x': x})
    df1.index = pd.to_datetime(df1.index)
    state_means = functions.KalmanFilterRegression(functions.KalmanFilterAverage(x), functions.KalmanFilterAverage(y))
    df1['hr'] = abs(state_means[:, 0])  # Hedge Ratio
    df1['spread'] = df1.y - (df1.x * df1.hr)
    test_spread[f"{pair1_name}_{pair2_name}"] = df1['spread']
test_spread = pd.DataFrame(test_spread)
test_spread.index = pd.to_datetime(test_data.index)

#train_spread.to_csv('train_spread.csv', index=True)
#test_spread.to_csv('test_spread.csv', index=True)

# Spread base csv
'''
train_spread_base = pd.DataFrame()
test_spread_base = pd.DataFrame()
for i, pairs in enumerate(cointegration_pairs):
    x = train_df[pairs[0]]
    y = train_df[pairs[1]]
    state_means = KalmanFilterRegression(KalmanFilterAverage(x), KalmanFilterAverage(y))
    hedge_ratio = abs(state_means[:, 0])
    train_spread_base[f"Pair {str(i+1)}"] = x * abs(hedge_ratio) + y

    x = test_df[pairs[0]]
    y = test_df[pairs[1]]
    state_means = KalmanFilterRegression(KalmanFilterAverage(x), KalmanFilterAverage(y))
    hedge_ratio = abs(state_means[:, 0])
    test_spread_base[f"Pair {str(i+1)}"] = x * abs(hedge_ratio) + y
train_spread_base.to_csv("train_spread_base.csv")
test_spread_base.to_csv("test_spread_base.csv")
'''
