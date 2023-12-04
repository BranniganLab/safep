# Import block
import numpy as np

from scipy.stats import norm

# Calculate the coefficient of determination:
def get_Rsq(X, Y, Yexpected):
    '''
    Calculate the coefficient of determination for arbitrary fits.
    Arguments: X (inputs), Y (experimental data), Yexpected (fitted or predicted data)
    '''
    residuals = Y-Yexpected
    SSres = np.sum(residuals**2)
    SStot = np.sum((X-np.mean(X))**2)
    R = 1-SSres/SStot
    return R
 
#Wrappers
def cum_Fn(x, m, s):
    '''
    Wrapper for the normal cumulative density function
    '''
    r = norm.cdf(x, m, s)
    return r

def pdf_Fn(x,m,s):
    '''
    Wrapper for the normal probability density function
    '''
    r = norm.pdf(x,m,s)
    return r

def moving_average(x, w):
    '''
    Generate a moving average over x with window width w
    Arguments: x (data), w (window width)
    Returns: a moving window average
    '''
    return np.convolve(x, np.ones(w), 'same') / w