from scipy.special import erfc
from scipy.optimize import curve_fit as scipyFit
from scipy.stats import skew

# Calculate the coefficient of determination:
def GetRsq(X, Y, Yexpected):
    residuals = Y-Yexpected
    SSres = np.sum(residuals**2)
    SStot = np.sum((X-np.mean(X))**2)
    R = 1-SSres/SStot
    R
 
#Wrapper for fitting the normal CDF 
def cumFn(x, m, s):
    r = norm.cdf(x, m, s)
    return r

def pdfFn(x,m,s):
    r = norm.pdf(x,m,s)
    return r