#%%
import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.analyzers as btanalyzers
import yfinance as yf
import datetime as dt

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from pykalman import KalmanFilter

import matplotlib.pyplot as plt

#%%
cointegration_pairs = []
pair_df = pd.read_csv("output/pair_df.csv", header = 0, index_col = 0)
for i in range(pair_df.shape[0]):
    cointegration_pairs.append([pair_df.iloc[i, 0], pair_df.iloc[i, 1]])

#%%
test_spread = pd.read_csv("output/test_spread.csv", header = 0, index_col = 0)
test_z_score = pd.read_csv("output/test_z_score.csv", header = 0, index_col = 0)
z_score_entry = pd.read_csv("output/z_score_optim.csv", header = 0, index_col = 0)

signal_df1 = []
entryZscore, exitZscore =  [0.8, -0.05]

for i in range(len(cointegration_pairs)):
    
    z_score = test_z_score.iloc[:, i]
    spread = test_spread.iloc[:, i]
    
    df = pd.DataFrame({"Z_score": z_score, "Spread": spread})
    df['long entry'] = (df["Z_score"] < (-entryZscore)) & (df["Z_score"].shift(1) > (-entryZscore))
    df['long exit'] = (df["Z_score"] > (-exitZscore)) & (df["Z_score"].shift(1) < (-exitZscore))
    df['short entry'] = (df["Z_score"] > (entryZscore)) & (df["Z_score"].shift(1) < (entryZscore))
    df['short exit'] = (df["Z_score"] < (exitZscore)) & (df["Z_score"].shift(1) > (exitZscore))
    
    signal_df1.append(df.dropna())
    
    
signal_df2 = []

for i in range(len(cointegration_pairs)):
    
    z_score = test_z_score.iloc[:, i]
    spread = test_spread.iloc[:, i]
    
    longEntry, longExit, shortEntry, shortExit = z_score_entry.iloc[i, 2:]
    
    df = pd.DataFrame({"Z_score": z_score, "Spread": spread})
    df['long entry'] = (df["Z_score"] < longEntry) & (df["Z_score"].shift(1) > longEntry)
    df['long exit'] = (df["Z_score"] > longExit) & (df["Z_score"].shift(1) < longExit)
    df['short entry'] = (df["Z_score"] > shortEntry) & (df["Z_score"].shift(1) < shortEntry)
    df['short exit'] = (df["Z_score"] < shortExit) & (df["Z_score"].shift(1) > shortExit)
    
    signal_df2.append(df.dropna())
    
datetime_list = []
for index in signal_df1[0].index:
    datetime_list.append(dt.datetime.strptime(index, "%Y-%m-%d").date())
    
weight_p = pd.read_csv("output/weight_p_df.csv", header = 0, index_col = 0)
weight_t = pd.read_csv("output/weight_t_df.csv", header = 0, index_col = 0)
    
#%%
class HLOCV_csv(btfeeds.GenericCSVData):

  params = (
    ('fromdate', dt.datetime(2019, 1, 1)),
    ('todate', dt.datetime(2023, 12, 31)),
    ('nullvalue', 0.0),
    ('dtformat', ("%Y-%m-%d")),

    ('datetime', 0),
    ('high', 2),
    ('low', 3),
    ('open', 1),
    ('close', 4),
    ('volume', 5),
    ('openinterest', -1)
)     


class Strat_Equal_Weight_Fixed_Entry(bt.Strategy):
   
    def __init__(self):       
        sizer = []
        for _ in range(len(cointegration_pairs)):
            sizer.append([1, 1])
        self.trading_size = sizer
        
        
    def log(self, txt, dt = None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    # def notify_order(self, order):
    #     if order.status in [order.Submitted, order.Accepted]:
    #         # Buy/Sell order submitted/accepted to/by broker - Nothing to do
    #         return

    #     # Check if an order has been completed
    #     # Attention: broker could reject order if not enough cash
    #     if order.status in [order.Completed]:
    #         if order.isbuy():
    #             self.log(
    #                 'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
    #                 (order.executed.price,
    #                  order.executed.value,
    #                  order.executed.comm))

    #             self.buyprice = order.executed.price
    #             self.buycomm = order.executed.comm
    #         else:  # Sell
    #             self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
    #                      (order.executed.price,
    #                       order.executed.value,
    #                       order.executed.comm))

    #         self.bar_executed = len(self)

    #     elif order.status in [order.Canceled, order.Margin, order.Rejected]:
    #         self.log('Order Canceled/Margin/Rejected')

    #     self.order = None

    # def notify_trade(self, trade):
    #     if not trade.isclosed:
    #         return

    #     self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
    #              (trade.pnl, trade.pnlcomm))

    def next(self):
        
        if self.datas[0].datetime.date(0) in datetime_list:
            
           date_row = datetime_list.index(self.datas[0].datetime.date(0))
           
           for i in range(len(cointegration_pairs)):
               
                idx_0 = data_idx[cointegration_pairs[i][0]]
                idx_1 = data_idx[cointegration_pairs[i][1]]
                
                ratio = self.datas[idx_0].close[0] / self.datas[idx_1].close[0]
                size = self.broker.get_value() / len(cointegration_pairs) / self.datas[idx_0].close[0]
               
                if signal_df1[i].iloc[date_row, 2]:
                    self.sell(data = self.datas[idx_0], size = size)
                    self.buy(data = self.datas[idx_1], size = size * ratio)
                    self.trading_size[i] = [size, size * ratio]
                elif signal_df1[i].iloc[date_row, 3]:
                    self.buy(data = self.datas[idx_0], size = self.trading_size[i][0])
                    self.sell(data = self.datas[idx_1], size = self.trading_size[i][1])
                elif signal_df1[i].iloc[date_row, 4]:
                    self.buy(data = self.datas[idx_0], size = size)
                    self.sell(data = self.datas[idx_1], size = size * ratio)
                    self.trading_size[i] = [size, size * ratio]
                elif signal_df1[i].iloc[date_row, 5]:
                    self.sell(data = self.datas[idx_0], size = self.trading_size[i][0])
                    self.buy(data = self.datas[idx_1], size = self.trading_size[i][1])

                    
                    
class Strat_Equal_Weight_Optimized_Entry(bt.Strategy):
   
    def __init__(self):       
        sizer = []
        for _ in range(len(cointegration_pairs)):
            sizer.append([1, 1])
        self.trading_size = sizer
        
        
    def log(self, txt, dt = None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    # def notify_order(self, order):
    #     if order.status in [order.Submitted, order.Accepted]:
    #         # Buy/Sell order submitted/accepted to/by broker - Nothing to do
    #         return

    #     # Check if an order has been completed
    #     # Attention: broker could reject order if not enough cash
    #     if order.status in [order.Completed]:
    #         if order.isbuy():
    #             self.log(
    #                 'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
    #                 (order.executed.price,
    #                  order.executed.value,
    #                  order.executed.comm))

    #             self.buyprice = order.executed.price
    #             self.buycomm = order.executed.comm
    #         else:  # Sell
    #             self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
    #                      (order.executed.price,
    #                       order.executed.value,
    #                       order.executed.comm))

    #         self.bar_executed = len(self)

    #     elif order.status in [order.Canceled, order.Margin, order.Rejected]:
    #         self.log('Order Canceled/Margin/Rejected')

    #     self.order = None

    # def notify_trade(self, trade):
    #     if not trade.isclosed:
    #         return

    #     self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
    #              (trade.pnl, trade.pnlcomm))

    def next(self):
        
        if self.datas[0].datetime.date(0) in datetime_list:
            
           date_row = datetime_list.index(self.datas[0].datetime.date(0))
           
           for i in range(len(cointegration_pairs)):
               
                idx_0 = data_idx[cointegration_pairs[i][0]]
                idx_1 = data_idx[cointegration_pairs[i][1]]
                
                ratio = self.datas[idx_0].close[0] / self.datas[idx_1].close[0]
                size = self.broker.get_value() / len(cointegration_pairs) / self.datas[idx_0].close[0]
               
                if signal_df2[i].iloc[date_row, 2]:
                    self.sell(data = self.datas[idx_0], size = size)
                    self.buy(data = self.datas[idx_1], size = size * ratio)
                    self.trading_size[i] = [size, size * ratio]
                elif signal_df2[i].iloc[date_row, 3]:
                    self.buy(data = self.datas[idx_0], size = self.trading_size[i][0])
                    self.sell(data = self.datas[idx_1], size = self.trading_size[i][1])
                elif signal_df2[i].iloc[date_row, 4]:
                    self.buy(data = self.datas[idx_0], size = size)
                    self.sell(data = self.datas[idx_1], size = size * ratio)
                    self.trading_size[i] = [size, size * ratio]
                elif signal_df2[i].iloc[date_row, 5]:
                    self.sell(data = self.datas[idx_0], size = self.trading_size[i][0])
                    self.buy(data = self.datas[idx_1], size = self.trading_size[i][1])


class Strat_Optimized_Weight_Optimized_Entry(bt.Strategy):
   
    def __init__(self):       
        sizer = []
        for _ in range(len(cointegration_pairs)):
            sizer.append([1, 1])
        self.trading_size = sizer
        
        
    def log(self, txt, dt = None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    # def notify_order(self, order):
    #     if order.status in [order.Submitted, order.Accepted]:
    #         # Buy/Sell order submitted/accepted to/by broker - Nothing to do
    #         return

    #     # Check if an order has been completed
    #     # Attention: broker could reject order if not enough cash
    #     if order.status in [order.Completed]:
    #         if order.isbuy():
    #             self.log(
    #                 'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
    #                 (order.executed.price,
    #                  order.executed.value,
    #                  order.executed.comm))

    #             self.buyprice = order.executed.price
    #             self.buycomm = order.executed.comm
    #         else:  # Sell
    #             self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
    #                      (order.executed.price,
    #                       order.executed.value,
    #                       order.executed.comm))

    #         self.bar_executed = len(self)

    #     elif order.status in [order.Canceled, order.Margin, order.Rejected]:
    #         self.log('Order Canceled/Margin/Rejected')

    #     self.order = None

    # def notify_trade(self, trade):
    #     if not trade.isclosed:
    #         return

    #     self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
    #              (trade.pnl, trade.pnlcomm))

    def next(self):
        
        if self.datas[0].datetime.date(0) in datetime_list:
            
           date_row = datetime_list.index(self.datas[0].datetime.date(0))
           
           for i in range(len(cointegration_pairs)):
               
                idx_0 = data_idx[cointegration_pairs[i][0]]
                idx_1 = data_idx[cointegration_pairs[i][1]]
                
                if np.isnan(weight_p.iloc[date_row, i]):
                    weight = 1 / len(cointegration_pairs)
                else:
                    weight = weight_p.iloc[date_row, i]
                
                ratio = self.datas[idx_0].close[0] / self.datas[idx_1].close[0]
                size = self.broker.get_value() * weight / self.datas[idx_0].close[0]
               
                if signal_df2[i].iloc[date_row, 2]:
                    self.sell(data = self.datas[idx_0], size = size)
                    self.buy(data = self.datas[idx_1], size = size * ratio)
                    self.trading_size[i] = [size, size * ratio]
                elif signal_df2[i].iloc[date_row, 3]:
                    self.buy(data = self.datas[idx_0], size = self.trading_size[i][0])
                    self.sell(data = self.datas[idx_1], size = self.trading_size[i][1])
                elif signal_df2[i].iloc[date_row, 4]:
                    self.buy(data = self.datas[idx_0], size = size)
                    self.sell(data = self.datas[idx_1], size = size * ratio)
                    self.trading_size[i] = [size, size * ratio]
                elif signal_df2[i].iloc[date_row, 5]:
                    self.sell(data = self.datas[idx_0], size = self.trading_size[i][0])
                    self.buy(data = self.datas[idx_1], size = self.trading_size[i][1])
        
#%%
portfolio_starting_value = pow(10, 9)

cerebro = bt.Cerebro()
cerebro.broker.setcash(portfolio_starting_value)
cerebro.broker.setcommission(commission = 0.001)  # 0.1% transaction cost 

data_idx = {}
for pairs in cointegration_pairs:
    for ticker in pairs:
        if ticker not in list(data_idx.keys()):
            data = HLOCV_csv(dataname =  f"data/{ticker}.csv")
            data.plotinfo.plot = False
            cerebro.adddata(data, name = ticker)
            data_idx[ticker] = len(data_idx)
            
print("Equal Weight & Fixed Entry")
cerebro.addstrategy(Strat_Equal_Weight_Fixed_Entry)  


cerebro.addsizer(bt.sizers.FixedSize, stake = 100000)          
cerebro.addobserver(bt.observers.DrawDown)
cerebro.addanalyzer(btanalyzers.AnnualReturn, _name = "annual_return")
cerebro.addanalyzer(btanalyzers.SharpeRatio, _name = "mysharpe")
cerebro.addanalyzer(btanalyzers.DrawDown, _name = "drawdown")
cerebro.addanalyzer(btanalyzers.SQN, _name = "sqn")
            
print("Portfolio starting value:", cerebro.broker.getvalue())
thestrats = cerebro.run()
print("Portfolio ending value:", cerebro.broker.getvalue())

thestrat = thestrats[0]
print('Annualized Return:', thestrat.analyzers.annual_return.get_analysis())
print('Sharpe Ratio:', thestrat.analyzers.mysharpe.get_analysis())
print('Maximum Drawdown:', thestrat.analyzers.drawdown.get_analysis())
print('SQN:', thestrat.analyzers.sqn.get_analysis())
print()

cerebro.plot()

# %%
portfolio_starting_value = pow(10, 9)

cerebro = bt.Cerebro()
cerebro.broker.setcash(portfolio_starting_value)
cerebro.broker.setcommission(commission = 0.001)  # 0.1% transaction cost 

data_idx = {}
for pairs in cointegration_pairs:
    for ticker in pairs:
        if ticker not in list(data_idx.keys()):
            data = HLOCV_csv(dataname =  f"data/{ticker}.csv")
            data.plotinfo.plot = False
            cerebro.adddata(data, name = ticker)
            data_idx[ticker] = len(data_idx)
            
print("Equal Weight & Optimized Entry")
cerebro.addstrategy(Strat_Equal_Weight_Optimized_Entry)  


cerebro.addsizer(bt.sizers.FixedSize, stake = 100000)          
cerebro.addobserver(bt.observers.DrawDown)
cerebro.addanalyzer(btanalyzers.AnnualReturn, _name = "annual_return")
cerebro.addanalyzer(btanalyzers.SharpeRatio, _name = "mysharpe")
cerebro.addanalyzer(btanalyzers.DrawDown, _name = "drawdown")
cerebro.addanalyzer(btanalyzers.SQN, _name = "sqn")
            
print("Portfolio starting value:", cerebro.broker.getvalue())
thestrats = cerebro.run()
print("Portfolio ending value:", cerebro.broker.getvalue())

thestrat = thestrats[0]
print('Annualized Return:', thestrat.analyzers.annual_return.get_analysis())
print('Sharpe Ratio:', thestrat.analyzers.mysharpe.get_analysis())
print('Maximum Drawdown:', thestrat.analyzers.drawdown.get_analysis())
print('SQN:', thestrat.analyzers.sqn.get_analysis())
print()

cerebro.plot()

#%%
portfolio_starting_value = pow(10, 9)

cerebro = bt.Cerebro()
cerebro.broker.setcash(portfolio_starting_value)
cerebro.broker.setcommission(commission = 0.001)  # 0.1% transaction cost 

data_idx = {}
for pairs in cointegration_pairs:
    for ticker in pairs:
        if ticker not in list(data_idx.keys()):
            # price[ticker].iloc[training_split:].to_csv(f"data/{ticker}.csv")
            data = HLOCV_csv(dataname =  f"data/{ticker}.csv")
            data.plotinfo.plot = False
            cerebro.adddata(data, name = ticker)
            data_idx[ticker] = len(data_idx)
            
print("Optimzied Weight & Optimized Entry")
cerebro.addstrategy(Strat_Optimized_Weight_Optimized_Entry)  

cerebro.addsizer(bt.sizers.FixedSize, stake = 100000)          
cerebro.addobserver(bt.observers.DrawDown)
cerebro.addanalyzer(btanalyzers.AnnualReturn, _name = "annual_return")
cerebro.addanalyzer(btanalyzers.SharpeRatio, _name = "mysharpe")
cerebro.addanalyzer(btanalyzers.DrawDown, _name = "drawdown")
cerebro.addanalyzer(btanalyzers.SQN, _name = "sqn")
            
print("Portfolio starting value:", cerebro.broker.getvalue())
thestrats = cerebro.run()
print("Portfolio ending value:", cerebro.broker.getvalue())

thestrat = thestrats[0]
print('Annualized Return:', thestrat.analyzers.annual_return.get_analysis())
print('Sharpe Ratio:', thestrat.analyzers.mysharpe.get_analysis())
print('Maximum Drawdown:', thestrat.analyzers.drawdown.get_analysis())
print('SQN:', thestrat.analyzers.sqn.get_analysis())
print()

cerebro.plot()
# %%
