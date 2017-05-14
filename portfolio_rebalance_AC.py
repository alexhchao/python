# PORTFOLIO REBALANCING
# 10-12-2016
import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt
from datetime import datetime

tickers = ['ITOT','MUB','IEFA','EEM','TLT','HDV','USMV']
start = datetime(2006,1,1)
end = datetime(2017,12,31)
stockRawData = web.DataReader(tickers, 'yahoo', start, end)
print stockRawData.to_frame()
print stockRawData
sliceKey = 'Adj Close'
adjCloseData = stockRawData.ix[sliceKey]
print adjCloseData
get_annual_returns(stockRawData.ix['Adj Close'])

get_annual_returns(stockRawData.ix['Close'])



def get_data_from_yahoo( tickers,start, end = datetime.now(),field = 'Adj Close'):
    """
    Function gets list of tickers, pulls data from yahoo
    Parameters:
    tickers --  list of tickers
    start -- start date
    end -- end date
    field --field such as Adj Close, Open, High, Low, Close, Volume
    """
    stockRawData = web.DataReader(tickers, 'yahoo', start, end)
    return stockRawData.ix[field] 
       
# ============

tickers = ['SPY','TLT']
price = get_data_from_yahoo(tickers,'1/1/2003')
price.tail()
convert_frequency(price,'BA').pct_change()

returns = price.pct_change()
returns.head()

# CALCULATE SHARES
# =====
capital = 1000
#weights = pd.Series([.50,.50])
weights = [.5,.5]
shares = returns.copy()
shares.ix[:,:] = np.NaN
shares.head()

shares.iloc[0] = map(lambda x: x*capital, weights) # WEIGHT * CAPITAL
shares.iloc[0]  = shares.iloc[0] / price.iloc[0] # DIVIDE BY PRICE TO GET # SHARES
shares = shares.ffill()
shares.head()
#IF THIS IS BUY AND HOLD, JUST DRAG DeprecationWarning
capital = shares*price
capital['portfolio'] = capital.apply(sum,axis=1)

capital.head(30)
#divide by price to get our weights
weights = capital.iloc[:,0:2].div(capital.portfolio, axis='index')
weights.head(30)

# BUY AND HOLD
def asset_alloc_strategy(price, returns, weights, freq='none', capital = 1000):
    """
    Run asset allocation strategy based on rebalance frequency
    price -- price series
    returns -- returns series
    weights -- one line vector e.g. [.5,.5]
    freq -- rebalance frequency
    capital -- capital to start with
    =======
    Returns: shares, weights, capital
    """

    shares = returns.copy()
    shares.ix[:,:] = np.NaN
    shares.head()

    shares.iloc[0] = map(lambda x: x*capital, weights) # WEIGHT * CAPITAL
    shares.iloc[0]  = shares.iloc[0] / price.iloc[0] # DIVIDE BY PRICE TO GET # SHARES
    shares = shares.ffill()
    shares.head()
    #IF THIS IS BUY AND HOLD, JUST DRAG DeprecationWarning
    capital = shares*price
    capital['portfolio'] = capital.apply(sum,axis=1)

    #divide by price to get our weights
    weights = capital.iloc[:,0:2].div(capital.portfolio, axis='index')
    return shares, weights, capital
    
# ===
price = price.dropna()
returns = price.pct_change()

weights, capital = asset_alloc_strategy(price, returns, [.10,.10,.20,.40,0.0,.10,.10])

get_stats(capital)

shares.head()




    