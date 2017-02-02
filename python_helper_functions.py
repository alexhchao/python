# PYTHON HELPER FUNCTIONS
# list of helper functions to help with portfolio optimization, data analysis, stock data pulling, etc
# Author: Alex H Chao
# ================================


import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt
from numpy.linalg import inv, qr
from datetime import datetime


def merge_xts(list_df,join='outer'):
    """
    Input: multiple xts series e.g. [df1,df2,df3], join = outer, inner, left
    Output: Returns joined dataframe
    """
    return pd.concat(list_df,join=join,axis=1)


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
    
def convert_frequency(data, freq='BM'):
    """
    convert frequency to monthly, annual, etc
    Parameters:
    data -- dataframe
    freq -- BM for monthly, BA for year end 
    """
    return data.resample(freq,how='last')

def calculate_sharpe(daily_price_series):
    """
    calculate sharpe from daily returns
    input: daily_price_series
    output: sharpe
    """
    daily_return_series = daily_price_series.pct_change()
    return (np.mean(daily_return_series)/ np.std(daily_return_series)) *np.sqrt(252)

def get_annual_returns(equity_series):
    return equity_series.resample('BA',how='last').pct_change()

def get_monthly_returns(equity_series):
    return equity_series.resample('BM',how='last').pct_change()


#found online - adds business days
def date_by_adding_business_days(from_date, add_days):
    business_days_to_add = add_days
    current_date = from_date
    while business_days_to_add > 0:
        current_date += datetime.timedelta(days=1)
        weekday = current_date.weekday()
        if weekday >= 5: # sunday = 6
            continue
        business_days_to_add -= 1
    return current_date
    
    
def contains_nulls(x):
    return x.isnull().values.any()
    
def year_frac(start, end):
    """
    Similar to excel's yearfrac function. Returns
    a year fraction between two dates (i.e. 1.53 years).
    Approximation using the average number of seconds
    in a year.
    Args:
        * start (datetime): start date
        * end (datetime): end date
    """
    if start > end:
        raise ValueError('start cannot be larger than end')

    # using days
    return (end-start).days / 365.0



def calc_cagr(prices):
    """
    Calculates the CAGR (compound annual growth rate) for a given price series.

    Args:
        * prices (pandas.Series): A Series of prices.
    Returns:
        * float -- cagr.

    """
    start = prices.index[0]
    end = prices.index[-1]
    num_years = (end-start).days / 365.0
    return (prices.ix[-1] / prices.ix[0]) ** (1 / num_years ) - 1
    
    
def max_drawdown(xs):
    i = np.argmax(np.maximum.accumulate(xs) - xs) # end of the period
    j = np.argmax(xs[:i]) # start of period
    output = dict()
    output['max_drawdown'] = xs[i]/xs[j] -1
    output['start'] = j
    output['end']   = i
    return output
    
def max_drawdown_absolute(returns):
    r = returns.add(1).cumprod()
    dd = r.div(r.cummax()).sub(1)
    mdd = dd.min()
    end = dd.argmin()
    start = r.loc[:end].argmax()
    return mdd, start, end
    

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
    
    
# BUY AND HOLD
def asset_alloc_strategy(price, returns, weights, freq='none', capital = 1000):
    """
    Run asset allocation strategy based on rebalance frequency
    * CURRENTLY ONLY BUY AND HOLD IS SUPPORTED
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

    shares.iloc[0] = map(lambda x: x*capital, weights) # WEIGHT * CAPITAL
    shares.iloc[0]  = shares.iloc[0] / price.iloc[0] # DIVIDE BY PRICE TO GET # SHARES
    shares = shares.ffill()
    #IF THIS IS BUY AND HOLD, JUST DRAG DeprecationWarning
    capital = shares*price
    capital['portfolio'] = capital.apply(sum,axis=1)

    #divide by price to get our weights
    weights = capital.div(capital.portfolio, axis='index')
    #exclude last column of 1s
    weights = weights.iloc[:,:-1]
    return weights, capital


# THIS INCLUDES REBALANCING

def asset_alloc_strategy_static(price, returns, initial_weights, freq='BM', capital = 1000):
    """
    Run asset allocation strategy based on rebalance frequency
    * STATIC WEIGHTS ONLY
    price -- price series
    returns -- returns series
    weights -- one line vector e.g. [.5,.5]
    freq -- rebalance frequency
       * W  = Weekly
       * BM = Monthly
       * BQ = Quarterly
       * BA = Annually
       * none 
    capital -- capital to start with
    =======
    Returns: shares, weights, capital
    """
    weights = returns.copy()
    weights.ix[:,:] = np.NaN # initialize 
    
    weights.iloc[0,:] = initial_weights
    
    if freq == 'none':
        rebalance_days = []
    else:
        rebalance_days = weights.resample(freq,how='last').index
    
    for i in range(1,len(weights)):
        if weights.index[i] in (rebalance_days):
            weights.iloc[i,:] = initial_weights #initialize portfolio
        else:
            weights.iloc[i,:] = weights.iloc[i-1,:]*(1+returns.iloc[i,:]) # calculate weights each day based on returns
            weights.iloc[i,:] = weights.iloc[i,:] / weights.iloc[i,:].sum() #we need to divide by sum to normalize to 1

    port_returns = (weights.shift(1)*returns).sum(axis=1) #IMPORTANT: NEED TO SHIFT WEIGHTS TO AVOID LOOKAHEAD
    port_equity = (1+port_returns).cumprod()
    
    return weights, port_equity

#CALC NEW WEIGHTS EVERY REBALANCE PERIOD
def dynamic_asset_alloc_strategy(price, returns, initial_weights, freq='BM', method = 'max_sharpe', lookback=252,capital = 1000):
    """
    Run asset allocation strategy based on rebalance frequency
    * DYNAMIC WEIGHTS AT EACH REBALANCE PERIOD
    price -- price series
    returns -- returns series
    weights -- one line vector e.g. [.5,.5]
    freq -- rebalance frequency
       * BM = Monthly
       * BQ = Quarterly
       * BA = Annually
       * none 
    method -- max_sharpe or min_variance
    capital -- capital to start with
    =======
    Returns: shares, weights, capital
    """
    weights = returns.copy()
    weights.ix[:,:] = np.NaN # initialize 
    
    weights.iloc[0,:] = initial_weights
    
    if freq == 'none':
        rebalance_days = []
    else:
        rebalance_days = weights.resample(freq,how='last').index
    
    for i in range(1,len(weights)):
        if weights.index[i] in (rebalance_days):
            # RECALC WEIGHTS, BASED ON SOME LOOKBACK
            # =====================================
            min_index = max(i-lookback,0)
            if method == 'min_variance':
                sigma = returns.iloc[min_index:i-1,:].cov()
                new_weights = get_min_var_weights(sigma)
            elif method == 'max_sharpe':
                new_weights = calc_mean_var_weights(returns.iloc[min_index:i-1,:])
            else: # USE MOMENTUM
                new_weights = get_momentum_ranks(price,i,lookback)
            # ==================================
            weights.iloc[i,:] = new_weights #rebalance portfolio
        else:
            weights.iloc[i,:] = weights.iloc[i-1,:]*(1+returns.iloc[i,:]) # calculate weights each day based on returns
            weights.iloc[i,:] = weights.iloc[i,:] / weights.iloc[i,:].sum() #we need to divide by sum to normalize to 1

    port_returns = (weights.shift(1)*returns).sum(axis=1)
    port_equity = (1+port_returns).cumprod()
    
    return weights, port_equity


def get_momentum_ranks(price,i,lookback):
    """
    helpder function to rank by past n-day momentum
    """
    min_index = max(i-lookback,0)
    momentum = price.iloc[i-1,:]/price.iloc[min_index,:] -1
    return momentum.rank()/sum(momentum.rank())

def rescale_equity(equity, level=1000):
    """
    rescales equity to given starting level (1000)
    """
    returns = equity.pct_change()
    returns.iloc[0] = 0
    return (1+returns).cumprod()*level


# PORTFOLIO OPTIMIZATION

# Closed Form Solution for Minimum Variance
def get_min_var_weights(sigma):
    """
    given Covariance matrix sigma, calculates the Min Variance weights
    """
    if sigma.shape[0] != sigma.shape[1]:
        print 'ERROR: COV MATRIX IS NOT SQUARE!'
        return
    return inv(sigma).dot(np.ones(len(sigma))) / np.ones(len(sigma)).dot(inv(sigma)).dot(np.ones(len(sigma)))

# GET VARIANCE OF A PORTFOLIO
def get_var_portfolio(x,sigma):
    """
    given Covariance matrix sigma (daily returns), weights, calculates the anualized variance of the portfolio
    sigma -- covariance matrix
    x -- weights vector
    """
    return x.T.dot(sigma).dot(x) 

def get_mu_portfolio(x,mu):
    """
    given weights and mu, calculates the anualized return of the portfolio
    mu -- expected returns vector
    x -- weights vector
    """ 
    return mu.T.dot(x)

def get_rolling_beta(my_y,my_x,my_window=252):
    """
    given x,y, runs ols regression on rolling window, outputs the beta coefficient series
    y -- your portfolio or asset
    x -- SPX
    window -- # days rolling period
    """ 
    model = pd.ols(y=my_y,x=my_x, window=my_window)
    return model.beta.x



def outut_to_excel(list_dfs, xls_path):
    """
    output a dictionary of dataframes into excel (each df in seperate sheets)
    list_dfs -- must be in dict format, not list
    xls_path -- filename
    window -- # days rolling period
    """ 
    writer = ExcelWriter(xls_path)
    for name in list_dfs:
        list_dfs[name].to_excel(writer,name)
    writer.save()
    
def drawdowns(equity_curve):
    """
    input: equity_curve
    output: MDD_start, MDD_end, MDD_duration, drawdown (in %)
    """
    i = np.argmax(np.maximum.accumulate(equity_curve.values) - equity_curve.values) # end of the period
    j = np.argmax(equity_curve.values[:i]) # start of period

    #drawdown=abs(100.0*(equity_curve[i]-equity_curve[j]))
    #AC MODIFIED
    drawdown = equity_curve[i]/equity_curve[j] -1
    DT=equity_curve.index.values

    start_dt=pd.to_datetime(str(DT[j]))
    MDD_start=start_dt.strftime ("%Y-%m-%d") 

    end_dt=pd.to_datetime(str(DT[i]))
    MDD_end=end_dt.strftime ("%Y-%m-%d") 

    NOW=pd.to_datetime(str(DT[-1]))
    NOW=NOW.strftime ("%Y-%m-%d")

    MDD_duration=np.busday_count(MDD_start, MDD_end)

    try:
        UW_dt=equity_curve[i:].loc[equity_curve[i:].values>=equity_curve[j]].index.values[0]
        UW_dt=pd.to_datetime(str(UW_dt))
        UW_dt=UW_dt.strftime ("%Y-%m-%d")
        UW_duration=np.busday_count(MDD_end, UW_dt)
    except:
        UW_dt="0000-00-00"
        UW_duration=np.busday_count(MDD_end, NOW)

    return MDD_start, MDD_end, MDD_duration, drawdown


def get_stats(equity_curve):
    """
    Get summary performance stats for a strategy or series
    input: equity_curve
    output: CAGR, Vol, Sharpe, Max Drawdown
    """
    stats = {}
    stats['CAGR'] = calc_cagr(equity_curve)
    stats['Vol'] = equity_curve.pct_change().std()*np.sqrt(252)
    stats['Sharpe'] = calculate_sharpe(equity_curve)
    MDD_start, MDD_end, MDD_duration, drawdown = drawdowns(equity_curve)
    stats['Max DD'] = drawdown
    return stats
    
    
def calc_mean_var_weights(returns, weight_bounds=(0., 1.),
                          rf=0.,
                          method='max_sharpe'):
    """
    Calculates the mean-variance weights given a DataFrame of returns.

    Args:
        * returns (DataFrame): Returns for multiple securities.
        * weight_bounds ((low, high)): Weigh limits for optimization.
        * rf (float): Risk-free rate used in utility calculation
        * method (str): minimization method
            Currently supported:
                - max_sharpe
                - min_var

    Returns:
        Series {col_name: weight}

    """
    def fitness(weights, exp_rets, covar, rf):
        # portfolio mean
        mean = sum(exp_rets * weights)
        # portfolio var
        var = np.dot(np.dot(weights, covar), weights)
        # util = sharpe ratio
        if method == 'max_sharpe':
            util = -(mean - rf) / np.sqrt(var)
        else:
            util = var
        # negative because we want to maximize and optimizer
        # minimizes metric
        return util

    n = len(returns.columns)

    # expected return defaults to mean return by default
    exp_rets = returns.mean()

    # calc covariance matrix
    covar = returns.cov()

    weights = np.ones([n]) / n
    bounds = [weight_bounds for i in range(n)]
    # sum of weights must be equal to 1
    constraints = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})
    optimized = minimize(fitness, weights, (exp_rets, covar, rf),
                         method='SLSQP', constraints=constraints,
                         bounds=bounds)
    # check if success
    if not optimized.success:
        raise Exception(optimized.message)

    # return weight vector
    return pd.Series({returns.columns[i]: optimized.x[i] for i in range(n)})
     
# HOW MANY NA's are there in each column?
def get_NAs(new):
    return new.apply(lambda x: len(x[x.isnull()]))
    
    
    
    