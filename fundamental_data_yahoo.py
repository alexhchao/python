# -*- coding: utf-8 -*-
"""
Scraping fundamental data from yahoo finance
https://pythonprogramming.net/price-to-book-ratio/
@author: alex
"""

import time
import urllib2
from urllib2 import urlopen


yahooKeyStats('AAPL')

stock = 'AAPL'
# =======================
https://finance.yahoo.com/quote/AAPL?p=AAPL

def yahooKeyStats(stock):
    try:
        sourceCode = urllib2.urlopen('https://finance.yahoo.com/quote/'+stock+'?p='+stock).read()
        pbr = sourceCode.split('PE Ratio (TTM)</span></td><td class="Ta(end) Fw(b)" data-test="PE_RATIO-value" data-reactid="422">">')[1].split('</td>')[0]
        print 'PE ratio:',stock,pbr

    except Exception,e:
        print 'failed in the main loop',str(e)
        
        
        
sourceCode = urllib2.urlopen('https://finance.yahoo.com/quote/'+stock+'?p='+stock).read()
pbr = sourceCode.split('data-test="PE_RATIO-value" data-reactid="422">')[1].split('</td>')[0]
print 'PE ratio:',stock,pbr

search_string = '<td class="Ta(end) Fw(b)" data-test="PE_RATIO-value" data-reactid="422">' 
in sourceCode