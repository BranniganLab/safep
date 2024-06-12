# Import block
import matplotlib.pyplot as plt

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


def get_exponential(u_nk):
    '''
    Get exponential estimation of the change in free energy.
    Arguments: u_nk in alchemlyb format
    Returns: l[ambdas], l_mid [lambda window midpoints], dG_f [forward estimates], dG_b [backward estimates]
    ''' 

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
    '''
    Extract key information from an alchemlyb.BAR object. Useful for plotting.
    Arguments: a fitted BAR object
    Returns: l[ambdas], l_mid [lambda window midpoints], f [the cumulative free energy], df [the per-window free energy], ddf [the per-window errors], errors [the cumulative error]
    '''
    
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
    '''
    Run both exponential and BAR estimators and return the results in tidy dataframes.
    Arguments: u_nk in the alchemlyb format, method of fitting (String: BAR, EXP, or both)
    Returns: perWindow estimates (including errors), cumulative estimates (including errors)
    '''
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
        expl, expmid, dG_fs, dG_bs = get_exponential(u_nk)

        cumulative[('EXP', 'ff')] = np.insert(np.cumsum(dG_fs),0,0)
        cumulative[('EXP', 'fb')] = np.insert(-np.cumsum(dG_bs),0,0)
        cumulative.index = expl 
        
        perWindow[('EXP','dG_f')] = dG_fs
        perWindow[('EXP','dG_b')] = dG_bs
        perWindow[('EXP', 'difference')] = np.array(dG_fs)+np.array(dG_bs)        
        perWindow.index = expmid
        
    
    perWindow.columns = pd.MultiIndex.from_tuples(perWindow.columns)
    perWindow = perWindow.fillna(0)
    cumulative.columns = pd.MultiIndex.from_tuples(cumulative.columns)
    cumulative = cumulative.fillna(0)
    
    return perWindow.copy(), cumulative.copy()
   
    
#Light-weight exponential estimator. Requires alternative parser.
def get_dG_from_data(data, temperature):
    from scipy.constants import R, calorie
    beta = 1/(R/(1000*calorie) * temperature) #So that the final result is in kcal/mol
    
    groups = data.groupby(level=0)
    dG=[]
    for name, group in groups:
        isUp = group.up
        dE = group.dE
        toAppend = [name, -1*np.log(np.mean(np.exp(-beta*dE[isUp]))), 1]
        dG.append(toAppend)
        toAppend=[name, -1*np.log(np.mean(np.exp(-beta*dE[~isUp]))), 0]
        dG.append(toAppend)
    
    dG = pd.DataFrame(dG, columns=["window", "dG", "up"])
    dG = dG.set_index('window')
    
    dG_f = dG.loc[dG.up==1] 
    dG_b = dG.loc[dG.up==0]

    dG_f = dG_f.dG.dropna()
    dG_b = dG_b.dG.dropna()

    return dG_f, dG_b