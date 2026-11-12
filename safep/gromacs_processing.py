# Import block

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from scipy.stats import linregress as lr
from scipy.stats import norm
from scipy.optimize import curve_fit as scipyFit

import pandas as pd

from alchemlyb.parsing import gmx
from alchemlyb.estimators import BAR
from alchemlyb.preprocessing import subsampling

from natsort import natsorted
from glob import glob  # file regexes

from .helpers import *
from .estimators import *

def gromacs_detect_equilibrium_u_nk(u_nk: pd.DataFrame):
    groups = u_nk.groupby("vdw-lambda")
    EQ = pd.DataFrame([])
    for key, group in groups:
        group = group[~group.index.duplicated(keep="first")]
        grp_sorted = group.sort_index(level="time")
        if key < 1:
            use_col = -1
        else:
            use_col = 0
        grp_series = group.dropna(axis=1).sort_index(axis=1).iloc[:, use_col]
        test = subsampling.equilibrium_detection(grp_sorted, grp_series)
        EQ = pd.concat([EQ, test])
    return EQ


def gromacs_decorrelate_u_nk(u_nk: pd.DataFrame, method="dE") -> pd.DataFrame:
    """Decorrelate each window without equilibrium detection.

    Args:
        u_nk (pd.DataFrame): the energies in the alchemlyb style
        method (str, optional): decorrelation method. See subsampling.decorrelate_u_nk.
            Defaults to "dE".

    Returns:
        pd.DataFrame: the decorrelated energies in the alchemlyb style
    """
    u_nk = u_nk.sort_index(axis=0, level=1).sort_index(axis=1)
    groups = u_nk.groupby("vdw-lambda")
    decorr = pd.DataFrame([])
    for key, group in groups:
        test = subsampling.decorrelate_u_nk(group, method)
        decorr = pd.concat([decorr, test])
    return decorr


def gromacs_read_and_process(xvgFiles, temperature, decorrelate, detectEQ):
    """
    Read a list of Gromacs xvg files and carry out any decorrelation or equilibrium detection
    Arguments: list of files to parse, temperature, decorrelate (boolean, whether or not to use alchemlyb's decorrelation functions), detectEQ (boolean, whether or not to use alchemlyb's equilibrium detection)
    Returns: u_nk_list
    """
    xvgFiles = natsorted(xvgFiles)
    u_nk_list = [extract_u_nk(xvg, T=temperature) for xvg in xvgFiles]

    if detectEQ and decorrelate:
        print("Warning: detecting equilibrium ALSO decorrelates the samples")
    
    if detectEQ:
        print("Detecting Equilibrium (includes decorrelating)")
        u_nk = gromacs_detect_equilibrium_u_nk(u_nk)                      
    elif decorrelate:
        print(f"Decorrelating samples")
        u_nk = decorrelate_u_nk(u_nk)

    return u_nk
