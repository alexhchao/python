
# PORTFOLIO OPTIMIZATION
import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt
from numpy.linalg import inv, qr
from datetime import datetime
from pandas import ExcelWriter
#import helper function file
execfile('python_helper_functions.py')

import os
os.getcwd()

# prepare to read in data from csv
#dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
#data = pd.read_csv('raw_data.csv',index_col='Date', parse_dates=[0], date_parser=dateparse)
data = pd.read_csv('raw_data.csv',parse_dates=[0])
# Data cleaning - fix index to be Date column
names = data.columns
data.rename(columns={'Unnamed: 0':'Date'}, inplace=True)
data.index = data['Date'] 
del data['Date']
returns = data['1/1/2002':]

# PROBLEM 1

def compute_cov_matrix(returns, min_obs=260, max_obs=2600, freq=1):
    """
    Input parameters: daily returns (TxN) indexed by date and asset, 
    minimum and maximum number of observations required to compute an estimate (e.g. minimum=260 days, maximum=2600 days), 
    frequency (1 for daily, 260 for annualized), used for returning daily COV or annualized COV
    Output: NxNxT covariance matrix, indexed by date and asset.  
    Each date should contain the estimated covariance matrix for that date, 
    using only information available as of that date.
    """
    #lets store COV matrices in a dict, one for each date 
    all_cov = {}
    #initialize our dict, normalize dates
    all_cov.fromkeys(returns.index.strftime("%m/%d/%Y"), None)
    
    #CHECK IF DATA SET IS ENOUGH
    if len(returns) < min_obs:
        raise ValueError('ERROR: Dataset does not contain enough observations')
    
    #start at the min obs index
    index_to_start = min_obs
    for i in range(index_to_start,len(returns)):
        # number of observations to compute COV matrix is between min and max. If above max, we start shifting it forward
        if i <= max_obs:
            #start from begining
            my_returns = returns.iloc[:i,:]
            this_cov = compute_single_cov_matrix(my_returns)
        else:
            #start from a rolling window
            my_returns = returns.iloc[(i-max_obs):i,:]
            this_cov = compute_cov_matrix(my_returns)
        # UPDATE DATE FORMAT FOR EASIER INDEXING
        d = returns.index[i].strftime("%m/%d/%Y")
        all_cov[d] = this_cov * np.sqrt(freq) # annualize cov matrix, if necessary
    return all_cov


all_cov = {}
#initialize our dict, normalize dates
all_cov.fromkeys(returns.index.strftime("%m/%d/%Y"), None)

#TEST RUN
all_cov = compute_cov_matrix(returns['1/1/2002':'1/1/2004'])
d = returns.index[i].strftime("%m/%d/%Y")

list(all_cov.keys())
all_cov['10/01/2015']

# ====
# HELPER FUNCTIONS 


def compute_single_cov_matrix(returns):
    """
    Helper function tocompute one covariance matrix given an input return series 
    To be called on every date 
    """
    num_assets = returns.shape[1]
    COV = np.zeros((num_assets, num_assets))
    for i in range(num_assets):
        xi = returns.iloc[:,i]
        COV[i,i] = xi.var()
        for j in range(i+1,num_assets):
            xj = returns.iloc[:,j]
            COV[i,j] = calc_cov(xi,xj)
            COV[j,i] = COV[i,j]
    return pd.DataFrame(COV,index=returns.columns, columns=returns.columns)
    
def calc_cov(xi, xj):
    """
    helper function to calculate covariance of two data series, returns a scalar
    """
    #return (xi*xj).mean() - xi.mean()*xj.mean()
    return np.corrcoef(xi,xj)[1,0]*xi.std()*xj.std()

# TEST PROBLEM !
cov_dict = compute_cov_matrix(returns.iloc[:300,:], min_obs=260, max_obs=2600, freq=1)
returns.iloc[:300,:].index
cov_dict['02/24/2003']

returns.iloc[:300,:].cov() - cov_dict['02/24/2003']
returns.iloc[:300,:].cov() - compute_single_cov_matrix(returns.iloc[:300,:])
cov2cor(returns.iloc[:300,:].cov()) - returns.iloc[:300,:].corr()
cov_mat = returns.iloc[:300,:].cov()
cov_mat = cov_dict
cor_mat, vol_mat = decomp(cov_dict)
vol_mat
cor_mat['02/05/2003']
vol_mat.set_index('Date')

cov2cor(cov_dict['02/24/2003'])

# ================
# PROBLEM 2
# ===================

def decomp(cov_matrix):
    """
    Input: cov matrix (NxNxT)
    Output: correlation matrix (NxNxT), volatility matrix (TxN)
    
    """
    cor_matrix = {}
    vol_matrix = pd.DataFrame()
    
    for d, cov_mat in cov_matrix.iteritems():
        this_cor, this_vol = cov2cor(cov_mat)
        cor_matrix[d] = this_cor
        this_vol = pd.DataFrame(this_vol).T
        this_vol['Date'] = d # keep track of dates
        vol_matrix = vol_matrix.append(this_vol)
        
    vol_matrix = vol_matrix.set_index('Date') # Set index to dates
    return cor_matrix, vol_matrix
    
    
# ==

def cov2cor(cov):
    """
    Helper Function: Converts a single covariance matrix into a correlation matrix
    Returns correlation matrix and volatility vetor
    EQUATION =
    CORREALTION MATRIX = DIAGONAL MATRIX WITH 1/sigma on diagonals * COVARIANCE MATRIX * DIAGONAL MATRIX WITH 1/sigma on diagonals
    """
    
    if cov.shape[0] != cov.shape[1]:
        raise ValueError('ERROR: COV MATRIX IS NOT SQUARE!')
    
    var_vector = np.diag(cov)
    vol_vector = np.sqrt(var_vector)
    one_over_vol = np.diag(1/vol_vector)
    cor_mat = one_over_vol.dot(cov).dot(one_over_vol)
    #return both correlation matri and vol vector
    return pd.DataFrame(cor_mat,index=cov.columns, columns=cov.columns), pd.DataFrame(vol_vector, index=cov.columns)


pd.DataFrame(vol_vector, index=cov.columns)

# ===============
# problem 3
vol_floor = vol_mat.iloc[0,:]
vol_floor = pd.DataFrame(vol_floor).T
type(vol_floor)

def vol_floor(cov_matrix, vol_floor):
    """
    Input: cov matrix (NxNxT),volatility_floor (1xN or TxN)
    Output: cov matrix (NxNxT) where volatilities >= volatility floor
    Basically, given a covariance matrix, ouput back a subset of the covariance matrix 
    which days when the volatilties have breached the volatility floor
    """
    # Assume vol_floor is 1xN vector
    
    #first, decomp the cov matrix
    cor_mat, vol_mat = decomp(cov_mat)
    
    #Get list of dates where volatilities >= vol_floor
    high_vols = vol_mat[vol_mat >= np.matrix(vol_floor)]
    
    dates = high_vols.index
    dates = pd.Series(dates)
    
    #create new cov_matrix
    cov_mat_new = {}
    
    #copy over values to a new cov matrix
    for d in dates:
        cov_mat_new[d] = cov_mat[d]
    
    return cov_mat_new

# ===

T = len(cov_matrix.keys())
# assume vol_floor is either 1 by N, or T by N
vol_floor_nrows = vol_floor.shape[0]
vol_floor_ncols = vol_floor.shape[1]

if vol_floor_nrows == 1:
    new_vol_floor = pd.concat([vol_floor]*T, ignore_index=True)
    
new_vol_floor.set_index(cov_matrix.keys(),inplace=True)
new_vol_floor.set_index(cov_matrix.keys())

# ======
this_cor, this_vol = cov2cor(cov_mat)
cor_matrix[d] = this_cor
this_vol = pd.DataFrame(this_vol).T
this_vol
vol_matrix = vol_matrix.append(this_vol)

var_vector = np.diag(cov)
vol_vector = np.sqrt(var_vector)
one_over_vol = np.diag(1/vol_vector)
cor_mat = one_over_vol.dot(cov).dot(one_over_vol)
pd.DataFrame(cor_mat,index=cov.columns, columns=cov.columns)

cor_mat.shape

cov




# =====
# ====
# PSUEDO CODE
# compute covariance matrix at any point
for(int i=0; i<n;i++)
   S(i,i) = var(xi);
   for(j = i+1; j<n; j++)
     S(i,j) = cov(xi, xj);
     S(j,i) = S(i,j);
   end
 end
 cov(xi, xj) = mean(xi.*xj) - mean(xi)*mean(xj);
 
 
def initialize_new_matrix(num_columns, num_rows):
    """
    helper function to initialize a new matrix
    """
    return [[0 for x in range(num_columns)] for y in range(num_rows)]

test = data_sub.iloc[:260,:]
test.head()
returns = test

compute_cov_matrix(returns)
returns.cov() - compute_cov_matrix(returns)
calc_cov(xi,xj)
np.cov(xi,xj)

# ======

# assume all data starts 1/1/2002

data_sub = data['1/1/2002':]
data_sub.tail()
    
    
