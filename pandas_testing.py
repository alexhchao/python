
import datetime

import pandas as pd
import numpy as np
import pandas.io.data
from pandas import Series, DataFrame
pd.__version__

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('figure', figsize=(8, 7))
mpl.__version__

# load AAPL data from yahoo finance

aapl = pd.io.data.get_data_yahoo('AAPL',
                                 start=datetime.datetime(2014, 1, 1),
                                 end=datetime.datetime(2016, 1, 1))
aapl.head()
aapl.tail()

import os
os.getcwd()

#set working directory
os.chdir(r"C:\Users\alex\OneDrive\CMU MSCF\SIDE PROJECTS\PYTHON")
os.getcwd()

#output to csv
aapl.to_csv('aapl_ohlc.csv')

# input csv
df = pd.read_csv('aapl_ohlc.csv', index_col='Date', parse_dates=True)
df.head()
df.tail()
ts = df['Adj Close']
ts
type(ts)
ts.dtypes
#plot
ts.plot()
#what type of object is this?
type(aapl)
#get dimensions
aapl.shape
#get data types for each column
aapl.dtypes

aapl.index

ts['2012-01-01':'2012-12-31'].plot()

ts['2015':'2015'].plot()

aapl_returns = aapl.pct_change(1)
aapl_returns.plot()

data = aapl[['Adj Close']]
data['Returns'] = aapl_returns['Adj Close']

np.mean(data.Returns)
np.std(data.Returns)



#get annualized returns
# endpoints
def convert_frequency(data, freq='M'):
    return data.resample(freq).sum().sort_index(ascending=False)
import datetime

#NOW get SPX

spx = pd.io.data.get_data_yahoo('^GSPC',
                                 start=datetime.datetime(1990, 1, 1),
                                 end=datetime.datetime(2016, 9, 1))
spx.tail()
spx.sort_index['2016-09-01']
spx = spx[['Adj Close']]
spx_annual = convert_frequency(spx, 'A')
spx_annual_returns = spx_annual.pct_change(1)
spx_annual_returns
spx.resample('A')
spx.resample('M')
spx.resample('A').ohlc() 

spx.loc['2014-12-31']
spx.index[0]
spx[['Adj Close']].loc['2016'].plot()
# TRY PANDAS DATA READER
type(spx)

# Try backtester
import pip
! pip install bt
! pip install zipline

#pip.main(['install','--root','bt'])
pip.main(['install','bt'])
import bt

#what version of python do i have ?
import sys
sys.version_info







