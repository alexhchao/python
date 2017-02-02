! pip install tradingWithPython
import pandas as pd
import tradingWithPython as twp
assert twp.__version__ >= '0.0.12' , 'Please update your twp module '
import tradingWithPython.lib.yahooFinance as yahoo # yahoo finance module

price = yahoo.getHistoricData('SPY')['adj_close'][-2000:]

price.plot()

figsize(10,6)

# sumulate two trades
signal = pd.Series(index=price.index) #init signal series

# while setting values this way, make sure that the dates are actually in the index
signal[pd.datetime(2008,1,3)] = -10000 # go short 
signal[pd.datetime(2009,1,5)] = 10000 # go long 10k$ on this day
signal[pd.datetime(2013,1,8)] = 0 # close position


bt = twp.Backtest(price,signal,initialCash=0)
bt.plotTrades()

figure()
bt.pnl.plot()
title('pnl')

figure()
bt.data.plot()
title('all strategy data')