# RM 1 PROJECT

# FIX DATA ISSUES

# CONTINUOUS MISSING DATA -> INTERPOLATE with random normals with mean t_end minus t_start
# HOLIDAYS -> INTERPOLATE with random normals of entire return spy_series.csv
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import norm

os.getcwd()
os.chdir(r"C:\Users\alex\Dropbox\RM I Project")

dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y')

price = pd.read_csv('price_2.csv',index_col='Date', parse_dates=[0], date_parser=dateparse)
price = pd.read_csv('price_2.csv',index_col='Date', parse_dates=[0])

price.head()
price.dtypes # get data type for each column
price.index

price[price.isnull()]
type(price)
price.dtypes()
returns = price.pct_change()
returns.head()
data_to_fill = price.loc['03/01/2010':'03/17/2010','CDX']
# for indexing use loc or iloc

#CALC MEAN TO FILL
mean_to_fill = price.loc['03/18/2010','CDX'] / price.loc['02/26/2010','CDX'] -1
mean_to_fill = mean_to_fill / len(data_to_fill)
random_fill = np.random.normal(mean_to_fill, np.std(returns['CDX']),len(data_to_fill))
returns['CDX']['03/01/2010':'03/17/2010'] = random_fill
returns['CDX']['02/25/2010':'03/19/2010']

returns = returns.apply(lambda x: x.fillna( np.random.normal(x.mean(axis=0), x.std(axis=0) ) ),axis=0 )

returns.loc['2008-01']
returns.isnull().values.any()

#CONVERT RETURNS TO PRICE
returns.head()
new_price = (1+ returns).cumprod()
returns.to_csv('returns_na_filled.csv')

# toy example
df = pd.DataFrame({'A': [1,2,3,np.nan,4,5,6], 'B' : [7,8,9,np.nan,10,11,12]})
df.dtypes
means = df.fillna(df.mean(axis=0))
means = np.random.normal(df.mean(axis=0), df.std(axis=0) )
type(df.mean(axis=0))

#FILL NA VALUES WITH RANDOM NORMAL OF EACH COLUMNS MEAN AND STD
df.apply(lambda x: x.fillna( np.random.normal(x.mean(axis=0), x.std(axis=0) ) ),axis=0 )

x = df.loc[:,'A']

df.fillna()
df.mean(axis=0)

df.fillna(df.mean(axis=1), axis=1)


# IMPORT PNL ,
# CALC ValueError
# ==================
pnl = pd.read_csv('pnl_2.csv',index_col='Date', parse_dates=[0])
pnl.head()
pnl.shape
pnl.dtypes
pnl = pnl.ix[1:] # omit first row

#split into 10 day increments
df = pnl
loss = df.groupby(np.arange(len(df))/10).sum()
loss.tail()
var_95 = np.percentile(np.sort(loss['TOTAL_PNL']), 5)
loss.shape 

var_10_day = loss.apply(lambda x: np.percentile(x, 5))
var_10_day
loss.head()
np.perc
df.loc[::10,:]

type(var_10_day)




#colmeans
colmeans = np.mean(loss)
col_std = np.std(loss)
var_10_day_calculated = colmeans + norm.ppf(0.05)*col_std

pnl_2008 = pnl['2008-07':'2008-12'].apply(sum)
#EXPORT TO EXCEL
writer = pd.ExcelWriter('VAR_10_DAY.xlsx')
var_10_day.to_frame().to_excel(writer,'var_10_day')
var_10_day_calculated.to_frame().to_excel(writer,'var_10_day_calculated')
pnl_2008.to_frame().to_excel(writer,'PNL_2008')
loss.to_excel(writer,'ten_day_loss')
writer.save()


pnl_2008.to_csv('2008_PNL.csv')



# MERGE ABS
abs_1 = pd.read_csv('ABSHE_1.csv',index_col='Date', parse_dates=[0])
abs_2 = pd.read_csv('ABSHE_2.csv',index_col='Date', parse_dates=[0])
abs_1.columns = ['ABSHE']
abs_2.columns = ['ABSHE']
abs_2 = abs_2.convert_objects(convert_numeric=True)
abs_combined.dtypes
abs_combined = abs_1.combine_first(abs_2)
abs_combined = abs_combined.convert_objects(convert_numeric=True)
abs_combined.plot()

abs_1['2011']
abs_2['2011']

#CHECK for NULLS 
abs_combined.isnull().values.any()

#FILL NA VALUES WITH RANDOM NORMAL OF EACH COLUMNS MEAN AND STD
abs_combined = abs_combined.apply(lambda x: x.fillna( np.random.normal(x.mean(axis=0), x.std(axis=0) ) ),axis=0 )

abs_combined.to_csv('ABS_combined_data.csv')

# EXAMPLE FROM BOOK p187
a = pd.Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],index=['f', 'e', 'd', 'c', 'b', 'a'])
b = pd.Series(np.arange(len(a), dtype=np.float64),index=['f', 'e', 'd', 'c', 'b', 'a'])

np.where(pd.isnull(a),b,a)
a.combine_first(b)

