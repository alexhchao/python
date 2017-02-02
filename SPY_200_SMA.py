# SPY - 200 SMA Model

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
equity_series.to_csv('spy_series.csv')

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
    

def max_sub_array(x):
    max_ending_here = max_so_far = 0
    for a in range(len(x)):
        max_ending_here = max(0, max_ending_here + x[a])
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far
    
max_sub_array(test)
for i in range(len(test)):
    print test[i]
    
test.index()

test = ffn.get('usmv', start='2016-09-01', end='2017-01-01')
returns = test.pct_change()
returns.iloc[1] 
test.plot()

calculate_max_drawdown(test)

drawdowns(equity_series)
equity_series['2009-03-09']/equity_series['2007-10-09']-1
equity_curve = equity_series
np.maximum.accumulate(equity_curve.values)
