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
from .estimators import *



def batchProcess(paths, RT, decorrelate, pattern, temperature, detectEQ):
    u_nks = {}
    affixes = {}

    #Read all
    for path in paths:
        print(f"Reading {path}")
        key = path.split('/')[-2]
        fepoutFiles = glob(path+'/'+pattern)
        u_nks[key], affix = readAndProcess(fepoutFiles, temperature, decorrelate, detectEQ)


    ls = {}
    l_mids = {}
    fs = {}
    dfs = {}
    ddfs = {}
    errorses = {}
    dG_fs = {}
    dG_bs = {}

    #do BAR fitting
    for key in u_nks:
        u_nk = u_nks[key]
        u_nk = u_nk.sort_index(level=1)
        bar = BAR()
        bar.fit(u_nk)
        ls[key], l_mids[key], fs[key], dfs[key], ddfs[key], errorses[key] = get_BAR(bar)
        
        expl, expmid, dG_fs[key], dG_bs[key] = get_EXP(u_nk)

    #Collect into dataframes - could be more pythonic but it works
    cumulative = pd.DataFrame()
    for key in ls:
        #cumulative[(key, 'l')] = ls[key]
        cumulative[(key, 'f')] = fs[key]
        cumulative[(key, 'errors')] = errorses[key]
    cumulative.columns = pd.MultiIndex.from_tuples(cumulative.columns)

    perWindow = pd.DataFrame()
    for key in ls:
        #perWindow[(key, 'l_mid')] = l_mids[key]
        perWindow[(key, 'df')] = dfs[key]
        perWindow[(key, 'ddf')] = ddfs[key]
        perWindow[(key, 'dG_f')] = dG_fs[key]
        perWindow[(key, 'dG_b')] = dG_bs[key]
    perWindow.columns = pd.MultiIndex.from_tuples(perWindow.columns)
    perWindow.index = l_mids[key]
    
    return u_nks, cumulative, perWindow, affix
    
def readAndProcess(fepoutFiles, temperature, decorrelate, detectEQ):
    from alchemlyb.preprocessing import subsampling
    
    fepoutFiles = natsorted(fepoutFiles)
    u_nk = namd.extract_u_nk(fepoutFiles, temperature)
    
    affix=""
    
    if decorrelate:
        print(f"Decorrelating samples. Flag='{decorrelate}'")
        method = 'dE'
        affix = f'{affix}_decorrelated_{method}'
        groups = u_nk.groupby('fep-lambda')
        decorr = pd.DataFrame([])
        for key, group in groups:
            test = subsampling.decorrelate_u_nk(group, method)
            decorr = pd.concat([decorr, test])
        u_nk = decorr
    else:
        affix = f'{affix}_unprocessed'
    
    if detectEQ:
        print("Detecting Equilibrium")
        affix = f"{affix}_AutoEquilibrium"
        groups = u_nk.groupby('fep-lambda')
        EQ = pd.DataFrame([])
        for key, group in groups:
            group = group[~group.index.duplicated(keep='first')]
            test = subsampling.equilibrium_detection(group, group.dropna(axis=1).iloc[:,-1])
            EQ = pd.concat([EQ, test])
        u_nk = EQ
    else:
        affix=f"{affix}_HardEquilibrium"

    return u_nk, affix
  
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w
    
# Subsamples a u_nk dataframe using percentiles [0-100] of data instead of absolute percents
def subSample(unkGrps, lowPct, hiPct):
    partial = []
    for key, group in unkGrps:
        idcs = group.index.get_level_values(0)
        
        lowBnd = np.percentile(idcs, lowPct, method='closest_observation')
        hiBnd = np.percentile(idcs, hiPct, method='closest_observation')
        mask = np.logical_and(idcs<=hiBnd, idcs>=lowBnd) 
        sample = group.loc[mask]
        if len(sample)==0:
            print(f"ERROR: no samples in window {key}")
            print(f"Upper bound: {hiBnd}\nLower bound: {lowBnd}")
            raise
            
        partial.append(sample)

    partial = pd.concat(partial)
    
    return partial

# altConvergence splits the data into percentile blocks. Inspired by block averaging
def altConvergence(u_nk, nbins):
    groups = u_nk.groupby('fep-lambda')

    #return data_list
    
    forward = []
    forward_error = []
    backward = []
    backward_error = []
    num_points = nbins
    for i in range(1, num_points+1):
        # forward
        partial = subSample(groups, 100*(i-1)/num_points, 100*i/num_points)
        estimate = BAR().fit(partial)
        l, l_mid, f, df, ddf, errors = get_BAR(estimate)
        
        forward.append(f.iloc[-1])
        forward_error.append(errors[-1])

    return np.array(forward), np.array(forward_error)

def doConvergence(u_nk, tau=1, num_points=10):
    groups = u_nk.groupby('fep-lambda')

    #return data_list
    
    forward = []
    forward_error = []
    backward = []
    backward_error = []
    for i in range(1, num_points+1):
        # forward
        partial = subSample(groups, 0, 100*i/num_points)
        estimate = BAR().fit(partial)
        l, l_mid, f, df, ddf, errors = get_BAR(estimate)
        
        forward.append(f.iloc[-1])
        forward_error.append(errors[-1])
        
        partial = subSample(groups, 100*(1-i/num_points), 100)
        estimate = BAR().fit(partial)
        l, l_mid, f, df, ddf, errors = get_BAR(estimate)
        
        backward.append(f.iloc[-1])
        backward_error.append(errors[-1])

    return np.array(forward), np.array(forward_error), np.array(backward), np.array(backward_error)

#Functions for bootstrapping estimates and generating confidence intervals
def bootStrapEstimate(u_nk, estimator='BAR', iterations=100, schedule=[10,20,30,40,50,60,70,80,90,100]):
    groups = u_nk.groupby('fep-lambda')

    if estimator == 'EXP':
        dGfs = {}
        dGbs = {}
        alldGs = {}
    elif estimator == 'BAR':
        dGs = {}
        errs = {}
    else:
        raise ValueError(f"unknown estimator: {estimator}")

    for p in schedule:
        Fs = []
        Bs = []
        fs = []
        Gs = []
        #rs = []
        for i in np.arange(iterations):
            sampled = pd.DataFrame([])
            for key, group in groups:
                N = int(p*len(group)/100)
                if N < 1:
                    N=1
                rows = np.random.choice(len(group), size=N)
                test = group.iloc[rows,:]
                sampled = pd.concat([sampled, test])
            if estimator == 'EXP':
                l, l_mid, dG_f, dG_b = get_EXP(pd.DataFrame(sampled))
                F = np.sum(dG_f)
                B = np.sum(-dG_b)
                Fs.append(F)
                Bs.append(B)
                Gs.append(np.mean([F,B]))
            elif estimator == 'BAR':
                tmpBar = BAR()
                tmpBar.fit(sampled)
                l, l_mid, f, df, ddf, errors = get_BAR(tmpBar)
                fs.append(f.values[-1])
                #rs.append(errors[-1])

        if estimator == 'EXP':
            dGfs[p] = Fs
            dGbs[p] = Bs
            alldGs[p] = Gs
        else:
            dGs[p] = fs
            #errs[p] = rs

    if estimator == 'EXP':
        fwd = pd.DataFrame(dGfs).melt().copy()
        bwd = pd.DataFrame(dGbs).melt().copy()
        alldGs = pd.DataFrame(alldGs).melt().copy()
        return (alldGs, fwd, bwd)
    else:
        alldGs = pd.DataFrame(dGs).melt().copy()
        #allErrors = pd.DataFrame(errs).melt().copy()
        return alldGs

def getLimits(allSamples):
    groups = allSamples.groupby('variable')
    means = []
    errors = []
    for key, group in groups:
        means.append(np.mean(group.value))
        errors.append(np.std(group.value))

    upper = np.sum([[x*1 for x in errors],means], axis=0)
    lower = np.sum([[x*(-1) for x in errors],means], axis=0)
    
    return (upper, lower, means)

def getEmpiricalCI(allSamples, CI=0.95):
    groups = allSamples.groupby('variable')

    uppers=[]
    lowers=[]
    means=[]
    for key, group in groups:
        uppers.append(np.sort(group.value)[round(len(group)*CI)])
        lowers.append(np.sort(group.value)[round(len(group)*(1-CI))])
        means.append(np.mean(group.value))

    return (uppers, lowers, means)

# Estimate the probability density distribution from the moving slope of a CDF. i.e. using the values X and their cumulative density FX
def getMovingAveSlope(X,FX,window):
    slopes = []
    Xwindowed = sliding_window_view(X, window)
    FXwindowed = sliding_window_view(FX, window)
   
    for i in np.arange(len(Xwindowed)):
        Xwindow = Xwindowed[i]
        FXwindow = FXwindowed[i]
        result = lr(Xwindow, FXwindow)
        m = result.slope
        slopes.append(m)
    return slopes

#Calculate the PDF of the discrepancies
def getPDF(dG_f, dG_b, DiscrepancyFitting='LS', dx=0.01, binNum=20):
    diff = dG_f + np.array(dG_b)
    diff.sort()
    X = diff
    Y = np.arange(len(X))/len(X)

    #fit a normal distribution to the existing data
    if DiscrepancyFitting == 'LS':
        fitted = scipyFit(cumFn, X, Y)[0] #Fit norm.cdf to (X,Y)
    elif DiscrepancyFitting == 'ML':
        fitted = norm.fit(X) # fit a normal distribution to X
    else:
        raise("Error: Discrepancy fitting code not known. Acceptable values: ML (maximum likelihood) or LS (least squares)")
    discrepancies = dG_f + np.array(dG_b)

    pdfY, pdfX = np.histogram(discrepancies, bins=binNum, density=True)
    pdfX = (pdfX[1:]+pdfX[:-1])/2

    pdfXnorm  = np.arange(np.min(X), np.max(X), dx)
    pdfYnorm = norm.pdf(pdfXnorm, fitted[0], fitted[1])

    pdfYexpected = norm.pdf(pdfX, fitted[0], fitted[1])
           
    return X, Y, pdfX, pdfY, fitted, pdfXnorm, pdfYnorm, pdfYexpected

#Light-weight exponential estimator
def get_dG_fromData(data, temperature):
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
    
def u_nk_fromDF(data, temperature, eqTime, warnings=True):
    from scipy.constants import R, calorie
    beta = 1/(R/(1000*calorie) * temperature) #So that the final result is in kcal/mol
    u_nk = pd.pivot_table(data, index=["step", "FromLambda"], columns="ToLambda", values="dE")
    #u_nk = u_nk.sort_index(level=0).sort_index(axis='columns') #sort the data so it can be interpreted by the BAR estimator
    u_nk = u_nk*beta
    u_nk.index.names=['time', 'fep-lambda']
    u_nk.columns.names = ['']
    u_nk = u_nk.loc[u_nk.index.get_level_values('time')>=eqTime]

    
    #Shift and align values to be consistent with alchemlyb standards
    lambdas = list(set(u_nk.index.get_level_values(1)).union(set(u_nk.columns)))
    lambdas.sort()
    warns = set([])
            
    for L in lambdas:
        try:
            u_nk.loc[(slice(None), L), L] = 0
        except:
            if warnings:
                warns.add(L)
    
    prev = lambdas[0]
    for L in lambdas[1:]:
        try:
            u_nk.loc[(slice(None), L), prev] = u_nk.loc[(slice(None), L), prev].shift(1)
        except:
            if warnings:
                warns.add(L)
            
        prev = L
    
    if len(warns)>0:
        print(f"Warning: lambdas={warns} not found in indices/columns")
    u_nk = u_nk.dropna(thresh=2)
    u_nk = u_nk.sort_index(level=1).sort_index(axis='columns') #sort the data so it can be interpreted by the BAR estimator
    return u_nk
    



