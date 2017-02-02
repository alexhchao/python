#WES MKCINNY BOOK
import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt
#import ffn 

#READ FROM CSV
pd.read_csv('CBOE_option_writing_indices.csv', index_col='Date', parse_dates=True)
 
all_data = {}
for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']:
    all_data[ticker] = pd.get_data_yahoo(ticker, '1/1/2000', '1/1/2010')
price = DataFrame({tic: data['Adj Close'] for tic, data in all_data.iteritems()})
volume = DataFrame({tic: data['Volume'] for tic, data in all_data.iteritems()})

returns = price.pct_change()
returns.head()
returns.corr()
returns.cov()

pd.rolling_corr(returns['AAPL'], returns['IBM'], 250).plot()

# ROLLING CORRELATIONS

spy = web.get_data_yahoo('SPY', '2000-01-01')['Adj Close']
tlt = web.get_data_yahoo('TLT', '2000-01-01')['Adj Close']
spy_rets = spy.pct_change()
tlt_rets = tlt.pct_change()
pd.rolling_corr(spy_rets, tlt_rets, 66).plot()

model = pd.ols(y=spy_rets, x={'TLT': tlt_rets}, window=250)

# LOOK INSIDE AN OBJECT
plt.clf()
model.beta.plot()

model.t_stat.plot()

#see values of all methods

model.__dict__

type(model)
dir(model)
id(model)
getattr(model)
hasattr(model)
globals(model)
locals(model)
callable(model)

spy_sma = pd.rolling_mean(spy,window=200)
spy_sma.plot()

#plot 2 lines on same graph
plt.plot(spy)
plt.plot(spy_sma)
plt.show()

spy_data = pd.concat([spy, spy_rets, spy_sma, spy > spy_sma],axis=1)
spy_data.columns = ['SPY','RETS','SMA','SIGNAL']
spy_data.tail()
spy_data['SIGNAL'] = spy_data['SIGNAL'].apply(int)

#plot 2 plots seperately (subplots)

plt.figure(1)
plt.subplot(211)
plt.plot(spy_data['SPY'])
plt.plot(spy_data['SMA'])
plt.show()

plt.subplot(212)
plt.plot(spy_data['SIGNAL'])
plt.show()

#ZOOM INTO 2008

spy_data['2008']['SIGNAL'].plot()


#LAG THE SIGNAL

spy_data['SIGNAL_LAG'] = spy_data['SIGNAL'].shift(2) #get signal eod 1, trade eod 2
spy_data.tail(10)

#STRATEGY
spy_data['STRAT'] = spy_data['RETS']*spy_data['SIGNAL_LAG']

#cum prod

spy_data['STRAT_EQUITY'] = (1+ spy_data['STRAT']).cumprod()
spy_data['SPY_EQUITY'] = (1+ spy_data['RETS']).cumprod()

spy_data[['STRAT_EQUITY','SPY_EQUITY']].plot()

#CALC SHARPE
stats = {}
stats['sharpe'] = np.mean(spy_data['STRAT'])*252 / (np.std(spy_data['STRAT'])*np.sqrt(252))
stats['sharpe'] = calculate_sharpe(spy_data['STRAT_EQUITY'])
stats['sharpe_SPY'] = calculate_sharpe(spy_data['SPY'])

# MAX DrawDown
equity_series = spy_data['SPY']
calculate_max_drawdown(equity_series)
equity_curve.calc_cagr()
equity_curve.calc_stats()
equity_curve.calc_max_drawdown()
#output to csv
#equity_series.to_csv('spy_series.csv')

def calculate_max_drawdown(equity_series):
    returns = equity_series.pct_change()
    max_so_far = 0
    max_ending_here = 0
    for i in range(1,len(returns)):
        if max_ending_here + returns.iloc[i] < 0:
            max_ending_here += returns.iloc[i] # still losing money
        else:
            max_ending_here = 0 # positive, reset max
            
        if max_ending_here < max_so_far:
            max_so_far = max_ending_here # update total max
    return max_so_far
    
#lets rewrite the max drawdown function
np.maximum.accumulate
i = np.argmax(np.maximum.accumulate(xs) - xs) # end of the period
j = np.argmax(xs[:i]) # start of period
    
    
test = ffn.get('usmv', start='2016-09-01', end='2017-01-01')
test = web.get_data_yahoo('usmv', '1/1/2016', '1/1/2017')['Adj Close']
returns = test.pct_change()
returns.iloc[1] 
test.plot()

calculate_max_drawdown(test)
max_drawdown_absolute(returns)
max_drawdown(np.matrix(test))

equity_series['2009-03-09']/equity_series['2007-10-09']-1
equity_curve = equity_series
np.maximum.accumulate(equity_curve.values)




# GROUP BY 
data = ffn.get('agg,tlt,spy,eem,efa,hyd,usmv,^GSPC', start='2010-01-01', end='2017-01-01')
print data.tail()
returns = data.pct_change()

# ANNUAL CORRELATIONS
# calculate correlations, split by year
spx_corr = lambda x: x.corrwith(x['gspc'])
by_year = returns.groupby(lambda x: x.year)
by_year.apply(spx_corr)

#sharpe by year
by_year.apply(lambda x: x.mean()/x.std()*np.sqrt(252))



# WHAT ABOUT SSO / UBT ?
sso_ubt = ffn.get('SSO,UBT',start='2010-01-01', end='2017-01-01')
sso_ubt.tail()

sso_ubt_rets = sso_ubt.pct_change()
sso_ubt_rets.corr()
by_year = sso_ubt_rets.groupby(lambda x: x.year)
by_year.apply(lambda x: x.corrwith(x['sso']))
by_year.apply(lambda x: x.mean()/x.std()*np.sqrt(252))


# RANDOM FUNCTIONS

#get column means
#colmeans
prc_rets.mean(axis=0)


# PANELS
# ============================================
# p 150
tickers = ['AAPL','GOOG','NFLX','AMZN']
pdata = pd.Panel(dict( (stk, web.get_data_yahoo(stk) ) for stk in tickers))

pdata = pdata.swapaxes('items','minor')

pdata['Close']

pdata['Adj Close','2010-01-04':,:]

stacked = pdata.ix[:,'5/30/2012':,:].to_frame()

