# -*- coding: utf-8 -*-
"""
IDEA: LEVERAGED LOW VOL ETF ?

"""
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

def merge_xts(df1,df2,join='outer'):
    """
    Input: 2 xts series, join = outer, inner, left
    Output: Returns joined dataframe
    """
    return pd.concat([df1,df2],join=join,axis=1)

def merge_xts(list_df,join='outer'):
    """
    Input: 2 xts series, join = outer, inner, left
    Output: Returns joined dataframe
    """
    return pd.concat(list_df,join=join,axis=1)

sptr_usmv_sphd = merge_xts([sptr,usmv,sphd],join='inner')
sptr_usmv_sphd.columns = ['sptr','usmv','sphd']

# READ CSV

usmv = pd.read_csv('usmv_index_history_1988.csv',index_col='Date', parse_dates=True)
sphd = pd.read_csv('SPHD_index_2006.csv',index_col='Date', parse_dates=True)

# MERGE 2 XTS!!!
pd.concat([spx,usmv],join='inner',axis=1)

sptr_usmv = merge_xts(sptr,usmv, join='inner')

sptr_usmv.columns = ['sptr','usmv']
sptr_usmv.head()
sptr_usmv.plot()
sptr_usmv_sphd.ix['2016-10-31':]

rescale_equity(sptr_usmv).plot()

rescale_equity(sptr_usmv_sphd.ix['2015-12-31':]).plot()

sptr_usmv.resample('BA',how='last').pct_change().mean()

mean_ret = sptr_usmv.pct_change().mean()*12
mean_vol =sptr_usmv.pct_change().std()*np.sqrt(12)

mean_ret / mean_vol

# drawdowns
drawdowns(sptr_usmv.ix[:,1])

drawdowns(sptr_usmv.ix[:,0])

# ================ GET SPTR DATA from YAHOO ====================================
tickers = ['^SP500TR']
start = '1988-01-01'
end = '2016-12-31'
stockRawData = web.DataReader(tickers, 'yahoo', start, end)
stockRawData.ix['Adj Close'] 

sptr = get_data_from_yahoo(tickers,'1/1/1988')
price = price.dropna()

returns = price.pct_change()

price.resample('BA',how='last').pct_change()

rescale_equity(price).plot()

np.mean(returns)*np.sqrt(252)/np.std(returns)

