# -*- coding: utf-8 -*-
"""
ANALYZE CBOE BUY WRITE INDICES
1/8/2017
@author: alex
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


data = pd.read_csv('CBOE_option_writing_indices.csv', index_col='Date', parse_dates=True)
data.head()

get_annual_returns(data)

data.resample('BA',how='last').pct_change()

data.dtypes

get_stats(data)
calc_cagr(data)
data.pct_change().std()*np.sqrt(252)