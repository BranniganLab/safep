# Import block
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from scipy.stats import linregress as lr
from scipy.stats import norm
from scipy.special import erfc
from scipy.optimize import curve_fit as scipyFit
from scipy.stats import skew

import pandas as pd

from alchemlyb.visualisation.dF_state import plot_dF_state
from alchemlyb.parsing import namd
from alchemlyb.estimators import BAR
from alchemlyb.visualisation.dF_state import plot_dF_state
from alchemlyb.visualisation import plot_convergence

import re
from tqdm import tqdm #for progress bars
from natsort import natsorted #for sorting "naturally" instead of alphabetically
from glob import glob #file regexes

# Calculate the coefficient of determination:
def GetRsq(X, Y, Yexpected):
    residuals = Y-Yexpected
    SSres = np.sum(residuals**2)
    SStot = np.sum((X-np.mean(X))**2)
    R = 1-SSres/SStot
    R
 
#Wrappers
def cumFn(x, m, s):
    r = norm.cdf(x, m, s)
    return r

def pdfFn(x,m,s):
    r = norm.pdf(x,m,s)
    return r