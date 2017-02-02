# -*- coding: utf-8 -*-
"""
HOW TO BUILD A PAIRS TRADING STRATEGY
1-19-2017
https://www.quantopian.com/posts/how-to-build-a-pairs-trading-strategy-on-quantopian

"""
import pylab
import numpy as np
import pandas as pd

import statsmodels
from statsmodels.tsa.stattools import coint
# just set the seed for the random number generator
np.random.seed(107)

import os
os.getcwd()
#os.chdir('c:\\Users\alex\OneDrive\CMU MSCF\SIDE PROJECTS\PYTHON')
os.chdir('c://Users/alex/OneDrive/CMU MSCF/SIDE PROJECTS/PYTHON')

#import helper function file
execfile('python_helper_functions.py')


import matplotlib.pyplot as plt

symbol_list = ['SCTY','RUN']

price = get_data_from_yahoo(symbol_list,'1/1/2006')
price = price.dropna()
price.head()

diff_series = (price.ix[:,0]-price.ix[:,1])

diff_series = S1 - S2
diff_series.plot()
plt.axhline(diff_series.mean(), color='black')

plt.scatter(price.ix[:,0],price.ix[:,1])

zscore(diff_series).plot()
plt.axhline(zscore(diff_series).mean(), color='black')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')

# https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing-Part-II


coint_stocks('SCTY','RUN')
price.RUN.pct_change().corr(price.SCTY.pct_change())
plot_scatter_series(price, 'SCTY','RUN')

def plot_scatter_series(df, ts1, ts2):
    plt.xlabel('%s Price ($)' % ts1)
    plt.ylabel('%s Price ($)' % ts2)
    plt.title('%s and %s Price Scatterplot' % (ts1, ts2))
    plt.scatter(df[ts1], df[ts2])
    plt.show()
    

# Calculate optimal hedge ratio "beta"
res = pd.ols(y=price['SCTY'], x=price["RUN"])
beta_hr = res.beta.x

# Calculate the residuals of the linear combination
price["res"] = price['SCTY'] - beta_hr*price["RUN"]

# Plot the residuals
plot_residuals(price)

# Calculate and output the CADF test on the residuals
cadf = ts.adfuller(price["res"])
pprint.pprint(cadf)

pairs = calculate_spread_zscore(price, ['SCTY','RUN'])

pairs = create_long_short_market_signals(pairs, ['SCTY','RUN'])

pairs.head()

pairs.spread.plot()
pairs.zscore.plot()


# ===========
# LET TEST A NIAVE BUY AND HOLD "PAIRS TRADE"

asset_alloc_strategy(price, returns, weights, freq='none', capital = 1000)


def calculate_spread_zscore(pairs, symbols, lookback=100):
    """Creates a hedge ratio between the two symbols by calculating
    a rolling linear regression with a defined lookback period. This
    is then used to create a z-score of the 'spread' between the two
    symbols based on a linear combination of the two."""
    
    # Use the pandas Ordinary Least Squares method to fit a rolling
    # linear regression between the two closing price time series
    print "Fitting the rolling Linear Regression..."
    model = pd.ols(y=pairs[symbols[0]], 
                   x=pairs[symbols[1]],
                   window=lookback)

    # Construct the hedge ratio and eliminate the first 
    # lookback-length empty/NaN period
    pairs['hedge_ratio'] = model.beta['x']
    pairs = pairs.dropna()

    # Create the spread and then a z-score of the spread
    print "Creating the spread/zscore columns..."
    pairs['spread'] = pairs[symbols[0]] - pairs['hedge_ratio']*pairs[symbols[1]]
    pairs['zscore'] = (pairs['spread'] - np.mean(pairs['spread']))/np.std(pairs['spread'])
    return pairs


def create_long_short_market_signals(pairs, symbols, 
                                     z_entry_threshold=2.0, 
                                     z_exit_threshold=1.0):
    """Create the entry/exit signals based on the exceeding of 
    z_enter_threshold for entering a position and falling below
    z_exit_threshold for exiting a position."""

    # Calculate when to be long, short and when to exit
    pairs['longs'] = (pairs['zscore'] <= -z_entry_threshold)*1.0
    pairs['shorts'] = (pairs['zscore'] >= z_entry_threshold)*1.0
    pairs['exits'] = (np.abs(pairs['zscore']) <= z_exit_threshold)*1.0

    # These signals are needed because we need to propagate a
    # position forward, i.e. we need to stay long if the zscore
    # threshold is less than z_entry_threshold by still greater
    # than z_exit_threshold, and vice versa for shorts.
    pairs['long_market'] = 0.0
    pairs['short_market'] = 0.0

    # These variables track whether to be long or short while
    # iterating through the bars
    long_market = 0
    short_market = 0

    # Calculates when to actually be "in" the market, i.e. to have a
    # long or short position, as well as when not to be.
    # Since this is using iterrows to loop over a dataframe, it will
    # be significantly less efficient than a vectorised operation,
    # i.e. slow!
    print "Calculating when to be in the market (long and short)..."
    for i, b in enumerate(pairs.iterrows()):
        # Calculate longs
        if b[1]['longs'] == 1.0:
            long_market = 1            
        # Calculate shorts
        if b[1]['shorts'] == 1.0:
            short_market = 1
        # Calculate exists
        if b[1]['exits'] == 1.0:
            long_market = 0
            short_market = 0
        # This directly assigns a 1 or 0 to the long_market/short_market
        # columns, such that the strategy knows when to actually stay in!
        pairs.ix[i]['long_market'] = long_market
        pairs.ix[i]['short_market'] = short_market
    return pairs

    
# FIT AN OLS TO GET THE REGRESSION COEFFICIENTS

lookback = 200
model = pd.ols(y=price.RUN, 
                   x=price.SCTY,
                   window_type='rolling',
                   window=lookback)
# display all variables contained within the "model" object
model.__dict__.keys()
model.beta.plot()

def zscore(series):
    return (series - series.mean()) / np.std(series)

# GIVEN 2 TICKERS, RETURNS T-STAT AND P_VALUE
def coint_stocks(ticker1, ticker2):
    """
    Given 2 tickers, returns t-stat and p-value of cointegration test
    """
    symbol_list = [ticker1,ticker2]
    price = get_data_from_yahoo(symbol_list,'1/1/2006')
    price = price.dropna()
    return coint(price.ix[:,0], price.ix[:,1])


# TRY FIRST WITH MADE UP DATA

X_returns = np.random.normal(0, 1, 100) # Generate the daily returns
# sum them and shift all the prices up into a reasonable range
X = pd.Series(np.cumsum(X_returns), name='X') + 50
X.plot()

some_noise = np.random.normal(0, 1, 100)
Y = X + 5 + some_noise
Y.name = 'Y'
pd.concat([X, Y], axis=1).plot()

(Y-X).plot()


score, pvalue, _ = coint(X,Y)
print pvalue
X.corr(Y)
