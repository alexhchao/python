# -*- coding: utf-8 -*-
"""
USING QUANDL

https://pythonprogramming.net/price-to-book-ratio/



"""

api_key = '-UVzwX5xnzsm6x68A1Qq'

import time
import urllib2
from urllib2 import urlopen

import quandl
mydata = quandl.get("FRED/GDP")
mydata.plot()

quandl.ApiConfig.api_key =api_key




zill_test = quandl.get("ZILL/C02564_A")

#NEW YORK PRICE TO RENT RATIO
nyc_prr = quandl.get("ZILL/C00001_PRR")
(1/nyc_prr).plot()

chi_prr = quandl.get("ZILL/C00003_PRR")
nyc_chi = merge_xts([nyc_prr,chi_prr])
nyc_chi.columns = ['nyc_prr','chi_prr']
nyc_chi.plot()


# ========================
city_codes = pd.read_csv('quandl_zillow_city_codes.csv')
#city_codes.iloc[:,3].split('|')
#PARSE THE LAST 2 COLUMNS
county_codes = pd.DataFrame(city_codes.iloc[:,3].str.split('|',1).tolist(), columns = ['County','Code'])
city_codes = pd.concat([city_codes, county_codes],axis=1)

all_prr = city_codes.Code.head(100).copy()

for i, code in enumerate(city_codes.Code.head(100)):
    get_str = 'ZILL/C' + code + '_PRR'
    prr = quandl.get(get_str).iloc[-1].Value # For now, just get the latest value
    all_prr[i] = prr

all_city_prr = pd.concat([city_codes.Code.head(100), all_prr],axis=1)
all_city_prr.columns = ['Code', 'PRR']
all_city_prr = pd.merge(all_city_prr, city_codes,on='Code')

all_city_prr.sort('PRR')

# ========================
def yahooKeyStats(stock):
    try:
        sourceCode = urllib2.urlopen('http://finance.yahoo.com/q/ks?s='+stock).read()
        pbr = sourceCode.split('Price/Book (mrq):</td><td class="yfnc_tabledata1">')[1].split('</td>')[0]
        print 'price to book ratio:',stock,pbr

    except Exception,e:
        print 'failed in the main loop',str(e)
		