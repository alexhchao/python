# -*- coding: utf-8 -*-
"""
DOW - ALL TIME HIGHS - WHAT HAPPENS USUALLY WHEN DOW REACHES AN ALL TIME HIGH ?

"""

tickers = ['^DJI']
start = datetime(1980,1,1)
end = datetime(2017,12,31)
stockRawData = web.DataReader(tickers, 'yahoo', start, end)
print stockRawData.to_frame()
print stockRawData
sliceKey = 'Adj Close'
adjCloseData = stockRawData.ix[sliceKey]
print adjCloseData
get_annual_returns(stockRawData.ix['Adj Close'])

get_annual_returns(stockRawData.ix['Close'])

dow = stockRawData.ix['Close'].copy()
dow['running_max'] = pd.expanding_max(dow)
dow['new_high'] = (dow['running_max'].pct_change()>0)*1

dow['new_high'].plot()

pd.concat([dow['new_high'],dow['new_high'].shift(1)])

test = dow.ix[1:10,0].copy()
test['LAG_1'] = dow.ix[:,0].shift(1)


# WHENEVER DOW REACHES AN ALL TIME HIGH, WHAT USUALLY HAPPENS FORWARD 30 DAYS? 3,6 MONTHS? 1 YEAR?


dow = merge_xts([dow, dow.ix[:,0].shift(-21)/ dow.ix[:,0] -1 ])
dow = merge_xts([dow, dow.ix[:,0].shift(-63)/ dow.ix[:,0] -1 ])
dow = merge_xts([dow, dow.ix[:,0].shift(-126)/ dow.ix[:,0] -1 ])
dow = merge_xts([dow, dow.ix[:,0].shift(-252)/ dow.ix[:,0] -1 ])
dow.columns = ['DJI','RUNNING_MAX','NEW_HIGH','FWD_1M','FWD_3M','FWD_6M','FWD_1Y']
all_time_highs = dow[dow['NEW_HIGH']==1]

all_time_highs['FWD_1Y'].plot(kind='hist')

dow.mean(axis=0)

dow.to_csv('DOW_ALL_TIME_HIGHS.csv')

# GET BREAKPOINTS OF DOW REACHING 15,000; 16,000; etc

dow['NEW_PEAK'] = np.floor(dow['RUNNING_MAX']/1000)
dow['NEW_PEAK'] = (dow['NEW_PEAK'].pct_change()>0)*1
new_peaks = dow[dow['NEW_PEAK'] ==1]
new_peaks['FWD_1Y'].plot(type='hist')


# PLOT AND ANNOTATE

