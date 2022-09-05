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

from .helpers import *


def get_EXP(u_nk):
    #the data frame is organized from index level 1 (fep-lambda) TO column
    #dG will be FROM column TO index
    groups = u_nk.groupby(level=1)
    dG=pd.DataFrame([])
    for name, group in groups:
        dG[name] = np.log(np.mean(np.exp(-1*group), axis=0))

    dG_f=np.diag(dG, k=1)
    dG_b=np.diag(dG, k=-1)

    l=dG.columns.to_list()
    l_mid = np.mean([l[1:],l[:-1]], axis=0)

    return l, l_mid, dG_f, dG_b
    
def get_BAR(bar):
    
    # Extract data for plotting
    states = bar.states_

    f = bar.delta_f_.iloc[0,:] # dataframe
    l = np.array([float(s) for s in states])
    # lambda midpoints for each window
    l_mid = 0.5*(l[1:] + l[:-1])

    # FE differences are off diagonal
    df = np.diag(bar.delta_f_, k=1)
    

    # error estimates are off diagonal
    ddf = np.array([bar.d_delta_f_.iloc[i, i+1] for i in range(len(states)-1)])

    # Accumulate errors as sum of squares
    errors = np.array([np.sqrt((ddf[:i]**2).sum()) for i in range(len(states))])
    
    
    return l, l_mid, f, df, ddf, errors
    
def do_estimation(u_nk, method='both'):
    u_nk = u_nk.sort_index(level=1)
    cumulative = pd.DataFrame()
    perWindow = pd.DataFrame()
    if method=='both' or method=='BAR':
        bar = BAR()
        bar.fit(u_nk)
        ls, l_mids, fs, dfs, ddfs, errors = get_BAR(bar)
        
        cumulative[('BAR', 'f')] = fs
        cumulative[('BAR', 'errors')] = errors
        cumulative.index = ls

        perWindow[('BAR','df')] = dfs
        perWindow[('BAR', 'ddf')] = ddfs
        perWindow.index = l_mids
        
    if method=='both' or method=='EXP':
        expl, expmid, dG_fs, dG_bs = get_EXP(u_nk)

        cumulative[('EXP', 'ff')] = np.insert(np.cumsum(dG_fs),0,0)
        cumulative[('EXP', 'fb')] = np.insert(-np.cumsum(dG_bs),0,0)
        cumulative.index = expl 
        
        perWindow[('EXP','dG_f')] = dG_fs
        perWindow[('EXP','dG_b')] = dG_bs
        perWindow[('EXP', 'difference')] = np.array(dG_fs)+np.array(dG_bs)        
        perWindow.index = expmid
        
    
    perWindow.columns = pd.MultiIndex.from_tuples(perWindow.columns)
    cumulative.columns = pd.MultiIndex.from_tuples(cumulative.columns)
    
    return perWindow.copy(), cumulative.copy()    
    
def get_dG(u_nk):
    #the data frame is organized from index level 1 (fep-lambda) TO column
    #dG will be FROM column TO index
    groups = u_nk.groupby(level=1)
    dG=pd.DataFrame([]) 
    for name, group in groups:
        dG[name] = np.log(np.mean(np.exp(-1*group)))
        dG = dG.copy() # this is actually faster than having a fragmented dataframe
        
    return dG