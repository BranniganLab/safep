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


def convergencePlot(theax, fs, ferr, bs, berr, fwdColor='#0072B2', bwdColor='#D55E00', lgndF=None, lgndB=None):
    if not lgndF:
        lgndF=fwdColor
        lgndB=bwdColor
        
    theax.errorbar(np.arange(len(fs))/len(fs)+0.1, fs, yerr=ferr, marker='o', linewidth=1, color=fwdColor, markerfacecolor='white', markeredgewidth=1, markeredgecolor=fwdColor, ms=5)
    theax.errorbar(np.arange(len(bs))/len(fs)+0.1, bs, yerr=berr, marker='o', linewidth=1, color=bwdColor, markerfacecolor='white', markeredgewidth=1, markeredgecolor=bwdColor, ms=5, linestyle='--')

    theax.xaxis.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    
    finalMean = fs[-1]
    theax.axhline(y= finalMean, linestyle='-.', color='gray')
    theax.plot(0, finalMean, linewidth=1, color=lgndF, label='Forward Time Sampling')
    theax.plot(0, finalMean, linewidth=1, color=lgndB, linestyle='--', label='Backward Time Sampling')
    
    return theax

def doConvPlot(ax, X, fs, ferr, fwdColor, label=None):
    ax.errorbar(X, fs, yerr=ferr, marker=None, linewidth=1, color=fwdColor, markerfacecolor='white', markeredgewidth=1, markeredgecolor=fwdColor, ms=5, label=label)
    return ax

#Cannonical convergence plot
def convergence_plot(u_nk, tau=1, units='kT', RT=0.59):
    forward, forward_error, backward, backward_error = doConvergence(u_nk, num_points=10)

    if units=='kcal/mol':
        forward = forward*RT
        forward_error = forward_error*RT
        backward = backward*RT
        backward_error = backward_error*RT

    ax = plot_convergence(forward, forward_error, backward, backward_error)

    if units=='kcal/mol':
        ax.set(ylabel=r'$\rm\Delta G$'+'\n(kcal/mol)')

    return plt.gca()
    
    
def fb_discrepancy_plot(l_mid, dG_f, dG_b):
    plt.vlines(l_mid, np.zeros(len(l_mid)), dG_f + np.array(dG_b), label="fwd - bwd", linewidth=3)
    plt.legend()
    plt.title('Fwd-bwd discrepancies by lambda')
    plt.xlabel('Lambda')
    plt.ylabel('Diff. in delta-G')
    #Return the figure
    return plt.gca()    

def fb_discrepancy_hist(dG_f, dG_b):
    plt.hist(dG_f + np.array(dG_b));
    plt.title('Distribution of fwd-bwd discrepancies')
    plt.xlabel('Difference in delta-G')
    plt.ylabel('Count')
    #return the figure
    return plt.gca()



























