# PROBLEM 1
# PORTFOLIO OPTIMIZATION
import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt
from numpy.linalg import inv, qr
from datetime import datetime

# ====
# HELPER FUNCTIONS 

def data_cleaning():
    data = pd.read_csv('raw_data.csv',parse_dates=[0])
    # Data cleaning - fix index to be Date column
    names = data.columns
    data.rename(columns={'Unnamed: 0':'Date'}, inplace=True)
    data.index = data['Date'] 
    del data['Date']
    returns = data['1/1/2002':]
    return returns
    
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
    return np.corrcoef(xi,xj)[1,0]*xi.std()*xj.std()

# PROBLEM 1

# PROVIDED BY CRAIG

# calculate unbiased cov (so use N-1 in denominator)
def cal_cov(X, freq):
    X -= X.mean(axis=0)
    N = X.shape[0]
    return (np.dot(X.T, X.conj()) / (N-1) * freq).squeeze()
    
    
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
            this_cov = compute_single_cov_matrix(my_returns)
        # UPDATE DATE FORMAT FOR EASIER INDEXING
        d = returns.index[i].strftime("%m/%d/%Y")
        all_cov[d] = this_cov * np.sqrt(freq) # annualize cov matrix, if necessary
    return all_cov

# PROBLEM 2a

def decomp(cov_matrix):
    """
    Input: cov matrix (NxNxT)
    Output: correlation matrix (NxNxT), volatility matrix (TxN)
    
    """
    cor_matrix = {}
    vol_matrix = pd.DataFrame()
    this_cor = pd.DataFrame()
    
    for d, cov_mat in cov_matrix.iteritems():
        this_cor, this_vol = cov2cor(cov_mat)
        cor_matrix[d] = this_cor
        this_vol = pd.DataFrame(this_vol).T
        this_vol['Date'] = d # keep track of dates
        vol_matrix = vol_matrix.append(this_vol)
        
    vol_matrix = vol_matrix.set_index('Date') # Set index to dates
    return cor_matrix, vol_matrix

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
    return pd.DataFrame(cor_mat,index=cov.columns, columns=cov.columns), pd.DataFrame(vol_vector, index=cov.columns)

# PROBLEM 2 b

def vol_floor(cov_matrix, volatility_floor):
    """
    Input: cov matrix (NxNxT),volatility_floor (1xN or TxN)
    Output: cov matrix (NxNxT) where volatilities >= volatility floor
    Basically, given a covariance matrix, ouput back a subset of the covariance matrix 
    which days when the volatilties have breached the volatility floor
    """
    # Assume vol_floor is 1xN vector
    #first, decomp the cov matrix
    cor_mat, vol_mat = decomp(cov_matrix)
    
    #Get list of dates where volatilities >= vol_floor
    high_vols = vol_mat[vol_mat >= np.matrix(volatility_floor)]
    dates = high_vols.index
    dates = pd.Series(dates)
    
    #create new cov_matrix
    cov_mat_new = {}
    
    #copy over values to a new cov matrix
    for d in dates:
        cov_mat_new[d] = cov_matrix[d]
    return cov_mat_new
    
if __name__ == "__main__":
    returns = data_cleaning()

    # Problem 1
    cov_matrix = compute_cov_matrix(returns, min_obs=260, max_obs=2600, freq=1)
    print cov_matrix['12/31/2003']
    
    # Problem 2a
    cor_matrix, vol_vector = decomp(cov_matrix)
    print cor_matrix['12/31/2003']
    
    # Problem 2b
    volatility_floor = [.002 for x in range(6)]
    cov_matrix_high_vol = vol_floor(cov_matrix, volatility_floor)
    print cov_matrix_high_vol['02/24/2003']











