# PORTFOLIO OPTIMIZATION
import pandas as pd
import pandas.io.data as web
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, qr
from datetime import datetime
from pandas import ExcelWriter
#import ffn 
import os
os.getcwd()
#os.chdir('c:\\Users\alex\OneDrive\CMU MSCF\SIDE PROJECTS\PYTHON')
os.chdir('c://Users/alex/OneDrive/CMU MSCF/SIDE PROJECTS/PYTHON')

#import helper function file
execfile('python_helper_functions.py')

tickers = ['SPY','IEFA','IEMG','TLT','GLD','USMV','SPHD','HDV','MTUM','EFAV','EEMV']
#tickers = ['SPY','IVV']
prc  = get_data_from_yahoo(tickers,'1/1/2010')
prc = prc.dropna()

#overall returns
(prc[-1:] / prc.iloc[0]) -1

prc_rets = prc.pct_change()
COV_MAT = prc_rets.cov()

# sharpe
np.mean(prc_rets)*np.sqrt(252)/np.std(prc_rets)
prc_rets.corr()
#get column means
prc['2016-11-08':] #whats happened since trump?
(prc[-1:] / prc.loc['2016-11-08']) -1
corr_mat = get_correlation_mat_through_time(prc_rets, expanding_window=False, lookback = 21)
corr_mat.loc['2016-11-08']

plt.plot(rescale_equity(prc['2016-01-01':]))

#reshape correlations to SPY
spy_cor = corr_mat.ix[:,'SPY',:].T
plt.plot(spy_cor)
spy_cor.tail(20)


def get_correlation_mat_through_time(prc_rets, expanding_window=False, lookback = 252):
    """
    Input: multi-dim return series
    output: correlation matrix through time
    Choose expanding window or fixed rolling window
    """
    corr_mat = {}
    if expanding_window:
        for i,date in enumerate(prc_rets.index):
            corr_mat[date] = prc_rets[:date].corr()
    else:
        for i,date in enumerate(prc_rets.index[lookback:]):
            corr_mat[date] = prc_rets[(i-lookback):i].corr()
    return pd.Panel(corr_mat)




#colmeans
mu = prc_rets.mean(axis=0)

# Closed Form Solution for Minimum Variance 
#x = inv(sigma).dot(np.ones(3)) / np.ones(3).dot(inv(sigma)).dot(np.ones(3))

x = get_min_var_weights(COV_MAT)

#x = np.matrix([1,0,0]).T
#var_portfolio = x.T.dot(sigma).dot(x) * np.sqrt(252)
var_portfolio = get_var_portfolio(x, COV_MAT)

annualized_vol = np.sqrt(var_portfolio)*np.sqrt(252)
annualized_vol 

#mu_portfolio = mu.T.dot(x)*252
mu_portfolio = get_mu_portfolio(x,mu)
mu_portfolio*252

get_annual_returns(prc).mean(axis=0)


# TEST IT OUT
# ==================

weights, capital = asset_alloc_strategy(prc, prc_rets, x,freq='BM')
capital.plot()
weights.plot()
get_stats(capital)
get_stats(prc.SPY)

get_rolling_beta(capital.pct_change(),compare_rets.SPY, 252).plot()


output = {}
output['price'] = prc
output['returns'] = prc_rets
output['weights'] = weights
output['capital'] = pd.DataFrame(capital)

outut_to_excel(output, 'SPY_TLT_GLD_REBALANCE.xlsx')


compare = pd.concat([prc['SPY'],strat_equity],axis=1)
compare.resample('BA',how='last').pct_change()

calculate_sharpe(compare)

compare_rets = compare.pct_change()
equity = prc.SPY

new_spx = rescale_equity(equity, 1000)
pd.concat([new_spx, strat_equity],axis=1, keys=['SPX','STRATEGY']).plot()


max_dd(capital.iloc[:,-1])

# get rolling beta
get_rolling_beta(compare_rets.portfolio,compare_rets.SPY, 252)

#get_rolling_beta(compare_rets.portfolio,compare_rets.SPY, window=252)

model = pd.ols(y=compare_rets.portfolio,x=compare_rets.SPY, window=252)
model.beta.x.plot()

# ===============
def max_dd(ser):
    max2here = pd.expanding_max(ser)
    dd2here = ser - max2here
    return dd2here.min()
    
    
#rescale
def rescale_equity(equity, level=1000):
    """
    rescales equity to given starting level (1000)
    """
    returns = equity.pct_change()
    returns[0] = 1
    return (1+returns).cumprod()*level
    
    



MDD_start, MDD_end, MDD_duration, drawdown = drawdowns(strat_equity)
plt.plot(strat_equity)
plt.plot([MDD_start, MDD_end], [strat_equity[MDD_start], strat_equity[MDD_end]], 'o', color='Red', markersize=10)

