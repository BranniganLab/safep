# Import block

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from scipy.stats import linregress as lr
from scipy.stats import norm
from scipy.optimize import curve_fit as scipyFit

import pandas as pd

from alchemlyb.parsing import namd
from alchemlyb.estimators import BAR
from alchemlyb.preprocessing import subsampling

from natsort import natsorted
from glob import glob  # file regexes

from .helpers import *
from .estimators import *


def batch_process(paths, RT, decorrelate, pattern, temperature, detectEQ):
    """
    Read and process all NAMD FEPout files that match a given pattern
    Arguments: paths (list of strings), RT (float), decorrelate (boolean, whether or not to use alchemlyb's decorrelation functions), pattern (to match), temperature, detectEQ (boolean, whether or not to use alchemlyb's equilibrium detection)
    Returns: lists of: u_nks, cumulative estimates, perWindow estimates, an affix (that describes the data generation protocol)
    """
    u_nks = {}
    affixes = {}

    # Read all
    for path in paths:
        print(f"Reading {path}")
        key = path.split("/")[-2]
        fepoutFiles = glob(path + "/" + pattern)
        u_nks[key], affix = read_and_process(
            fepoutFiles, temperature, decorrelate, detectEQ
        )

    ls = {}
    l_mids = {}
    fs = {}
    dfs = {}
    ddfs = {}
    errorses = {}
    dG_fs = {}
    dG_bs = {}

    # do BAR fitting
    for key in u_nks:
        u_nk = u_nks[key]
        u_nk = u_nk.sort_index(level=1)
        bar = BAR()
        bar.fit(u_nk)
        (
            ls[key],
            l_mids[key],
            fs[key],
            dfs[key],
            ddfs[key],
            errorses[key],
        ) = get_BAR(bar)

        expl, expmid, dG_fs[key], dG_bs[key] = get_exponential(u_nk)

    # Collect into dataframes - could be more pythonic but it works
    cumulative = pd.DataFrame()
    for key in ls:
        # cumulative[(key, 'l')] = ls[key]
        cumulative[(key, "f")] = fs[key]
        cumulative[(key, "errors")] = errorses[key]
    cumulative.columns = pd.MultiIndex.from_tuples(cumulative.columns)

    perWindow = pd.DataFrame()
    for key in ls:
        # perWindow[(key, 'l_mid')] = l_mids[key]
        perWindow[(key, "df")] = dfs[key]
        perWindow[(key, "ddf")] = ddfs[key]
        perWindow[(key, "dG_f")] = dG_fs[key]
        perWindow[(key, "dG_b")] = dG_bs[key]
    perWindow.columns = pd.MultiIndex.from_tuples(perWindow.columns)
    perWindow.index = l_mids[key]

    return u_nks, cumulative, perWindow, affix


def detect_equilibrium_u_nk(u_nk: pd.DataFrame):
    groups = u_nk.groupby("fep-lambda")
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


def decorrelate_u_nk(u_nk: pd.DataFrame, method="dE") -> pd.DataFrame:
    groups = u_nk.groupby("fep-lambda")
    decorr = pd.DataFrame([])
    for key, group in groups:
        test = subsampling.decorrelate_u_nk(group, method)
        decorr = pd.concat([decorr, test])
    return decorr


def get_n_samples(u_nk: pd.DataFrame) -> pd.Series:
    samples = pd.Series()
    grps = u_nk.groupby("fep-lambda")
    for l, g in grps:
        samples[l] = len(g.dropna(how="all"))
    return samples


def read_and_process(fepoutFiles, temperature, decorrelate, detectEQ):
    """
    Read NAMD fepout files for a single calculation and carry out any decorrelation or equilibrium detection
    Arguments: files to parse, temperature, decorrelate (boolean, whether or not to use alchemlyb's decorrelation functions), detectEQ (boolean, whether or not to use alchemlyb's equilibrium detection)
    Returns: u_nk
    """
    fepoutFiles = natsorted(fepoutFiles)
    u_nk = namd.extract_u_nk(fepoutFiles, temperature)
    
    if detectEQ and decorrelate:
    	print("Warning: detecting equilibrium ALSO decorrelates the samples")
    
    if detectEQ:
        print("Detecting Equilibrium (includes decorrelating)")
        u_nk = detect_equilibrium_u_nk(u_nk)
    elif decorrelate:
        print(f"Decorrelating samples")
        u_nk = decorrelate_u_nk(u_nk)

    return u_nk


#
def subsample(unkGrps, lowPct, hiPct):
    """
    Subsamples a u_nk dataframe using percentiles [0-100] of data instead of absolute percents.
    Arguments: unkGrps (u_nk grouped by fep-lambda), lowPct (lower percentile bound), hiPct (upper percentile bound)
    Returns: partial (a u_nk in which each window is subsampled between the lower and upper percentile bounds)
    """
    partial = []
    for key, group in unkGrps:
        idcs = group.index.get_level_values(0)

        lowBnd = np.percentile(idcs, lowPct, method="closest_observation")
        hiBnd = np.percentile(idcs, hiPct, method="closest_observation")
        mask = np.logical_and(idcs <= hiBnd, idcs >= lowBnd)
        sample = group.loc[mask]
        if len(sample) == 0:
            print(f"ERROR: no samples in window {key}")
            print(f"Upper bound: {hiBnd}\nLower bound: {lowBnd}")
            raise

        partial.append(sample)

    partial = pd.concat(partial)

    return partial


# altConvergence splits the data into percentile blocks. Inspired by block averaging
def alt_convergence(u_nk, nbins):
    """
    An alternative convergence calculation that uses percentile blocks instead of cumulative samples. Inspired by block averages. Uses the BAR estimator.
    Arguments: u_nk, nbins (number of blocks)
    Returns: forward estimates, forward errors
    """
    groups = u_nk.groupby("fep-lambda")

    forward = []
    forward_error = []
    backward = []
    backward_error = []
    num_points = nbins
    for i in range(1, num_points + 1):
        # forward
        partial = subsample(
            groups, 100 * (i - 1) / num_points, 100 * i / num_points
        )
        estimate = BAR().fit(partial)
        l, l_mid, f, df, ddf, errors = get_BAR(estimate)

        forward.append(f.iloc[-1])
        forward_error.append(errors[-1])

    return np.array(forward), np.array(forward_error)


def do_convergence(u_nk, tau=1, num_points=10):
    """
    Convergence calculation. Incrementally adds data from either the start or the end of each windows simulation and calculates the resulting change in free energy.
    Arguments: u_nk, tau (an error scaling factor), num_points (number of chunks)
    Returns: forward-sampled estimate (starting from t=start), forward-sampled error, backward-sampled estimate (from t=end), backward-sampled error
    """
    groups = u_nk.groupby("fep-lambda")

    forward = []
    forward_error = []
    backward = []
    backward_error = []
    for i in range(1, num_points + 1):
        # forward
        partial = subsample(groups, 0, 100 * i / num_points)
        estimate = BAR().fit(partial)
        l, l_mid, f, df, ddf, errors = get_BAR(estimate)

        forward.append(f.iloc[-1])
        forward_error.append(errors[-1])

        partial = subsample(groups, 100 * (1 - i / num_points), 100)
        estimate = BAR().fit(partial)
        l, l_mid, f, df, ddf, errors = get_BAR(estimate)

        backward.append(f.iloc[-1])
        backward_error.append(errors[-1])

    return (
        np.array(forward),
        np.array(forward_error),
        np.array(backward),
        np.array(backward_error),
    )

def do_per_lambda_convergence(u_nk):
    """
    Convergence calculation. Incrementally adds data from either the start or the end of each windows simulation and calculates the resulting change in free energy.
    Arguments: u_nk, tau (an error scaling factor), num_points (number of chunks)
    Returns: forward-sampled estimate (starting from t=start), forward-sampled error, backward-sampled estimate (from t=end), backward-sampled error
    """
    groups = u_nk.groupby("fep-lambda")

    start = 0
    midpoint = 50
    end = 100

    partial = subsample(groups, start, midpoint)
    first_half, _ = do_estimation(partial, method='BAR')

    partial = subsample(groups, midpoint, end)
    last_half, _ = do_estimation(partial, method='BAR')

    convergence = last_half-first_half
    convergence.loc[:,('BAR','ddf')] = [np.sqrt(fe**2+be**2) for fe, be in zip(first_half.BAR.ddf, last_half.BAR.ddf)]
    return convergence


# Functions for bootstrapping estimates and generating confidence intervals
def bootstrap_estimate(
    u_nk,
    estimator="BAR",
    iterations=100,
    schedule=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
):
    """
    Bootstrapped free energy estimates with variable sample sizes. E.g. if schedule=[10], each bootstrapped sample will only be 10% the size of the original
    Arguments: u_nk, estimator (BAR or EXP[onential]), iterations (number of bootstrapped datasets to generate for each sample size), schedule (percent data to sample)
    Returns: if estimator=='BAR', N estimates for each schedule value. if estimator=='EXP', N estimates of forward, backward, and the mean of the two for each schedule value
    """
    groups = u_nk.groupby("fep-lambda")

    if estimator == "EXP":
        dGfs = {}
        dGbs = {}
        alldGs = {}
    elif estimator == "BAR":
        dGs = {}
        errs = {}
    else:
        raise ValueError(f"unknown estimator: {estimator}")

    for p in schedule:
        Fs = []
        Bs = []
        fs = []
        Gs = []
        # rs = []
        for i in np.arange(iterations):
            sampled = pd.DataFrame([])
            for key, group in groups:
                N = int(p * len(group) / 100)
                if N < 1:
                    N = 1
                rows = np.random.choice(len(group), size=N)
                test = group.iloc[rows, :]
                sampled = pd.concat([sampled, test])
            if estimator == "EXP":
                l, l_mid, dG_f, dG_b = get_exponential(pd.DataFrame(sampled))
                F = np.sum(dG_f)
                B = np.sum(-dG_b)
                Fs.append(F)
                Bs.append(B)
                Gs.append(np.mean([F, B]))
            elif estimator == "BAR":
                tmpBar = BAR()
                tmpBar.fit(sampled)
                l, l_mid, f, df, ddf, errors = get_BAR(tmpBar)
                fs.append(f.values[-1])
                # rs.append(errors[-1])

        if estimator == "EXP":
            dGfs[p] = Fs
            dGbs[p] = Bs
            alldGs[p] = Gs
        else:
            dGs[p] = fs
            # errs[p] = rs

    if estimator == "EXP":
        fwd = pd.DataFrame(dGfs).melt().copy()
        bwd = pd.DataFrame(dGbs).melt().copy()
        alldGs = pd.DataFrame(alldGs).melt().copy()
        return (alldGs, fwd, bwd)
    else:
        alldGs = pd.DataFrame(dGs).melt().copy()
        # allErrors = pd.DataFrame(errs).melt().copy()
        return alldGs


def get_limits(allSamples):
    """
    Get the mean and +/- 1 standard deviation for each group in a matrix
    Arguments: allSamples (the dataframe)
    Returns: (upper bounds, lower bounds, means)
    """
    groups = allSamples.groupby("variable")
    means = []
    errors = []
    for key, group in groups:
        means.append(np.mean(group.value))
        errors.append(np.std(group.value))

    upper = np.sum([[x * 1 for x in errors], means], axis=0)
    lower = np.sum([[x * (-1) for x in errors], means], axis=0)

    return (upper, lower, means)


def get_empirical_CI(allSamples, CI=0.95):
    """
    Get empirical confidence intervals from a dataframe
    Arguments: allSamples (the dataframe), CI (the interval to determine)[0.95]
    Returns: (upper bounds, lower bounds, means)
    """
    groups = allSamples.groupby("variable")

    uppers = []
    lowers = []
    means = []
    for key, group in groups:
        uppers.append(np.sort(group.value)[round(len(group) * CI)])
        lowers.append(np.sort(group.value)[round(len(group) * (1 - CI))])
        means.append(np.mean(group.value))

    return (uppers, lowers, means)


def get_moving_ave_slope(X, FX, window):
    """
    Estimates the PDF from a moving window average of a CDF
    Arguments: X (inputs), FX (the CDF), window (width)
    Returns: slopes
    """
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


# Calculate the PDF of the discrepancies
def get_PDF(dG_f, dG_b, DiscrepancyFitting="LS", dx=0.01, binNum=20):
    """
    Estimate the PDF of the discrepancy data by fitting to a gaussian
    Arguments: Forward estimates, backward estimates, DiscrepancyFitting ('LS' least squares of 'ML' maximum likelihood), dx (sample width for fitted gaussian), binNum (number of bins for histogramming)
    Returns: (experimental) X, Y, pdfX, pdfY, fitted parameters, (fitted) pdfXnorm, pdfYnorm, pdfYexpected (for each experimental X)
    """
    diff = dG_f + np.array(dG_b)
    diff.sort()
    X = diff
    Y = np.arange(len(X)) / len(X)

    # fit a normal distribution to the existing data
    if DiscrepancyFitting == "LS":
        fitted = scipyFit(cumFn, X, Y)[0]  # Fit norm.cdf to (X,Y)
    elif DiscrepancyFitting == "ML":
        fitted = norm.fit(X)  # fit a normal distribution to X
    else:
        raise (
            "Error: Discrepancy fitting code not known. Acceptable values: ML (maximum likelihood) or LS (least squares)"
        )
    discrepancies = dG_f + np.array(dG_b)

    pdfY, pdfX = np.histogram(discrepancies, bins=binNum, density=True)
    pdfX = (pdfX[1:] + pdfX[:-1]) / 2

    pdfXnorm = np.arange(np.min(X), np.max(X), dx)
    pdfYnorm = norm.pdf(pdfXnorm, fitted[0], fitted[1])

    pdfYexpected = norm.pdf(pdfX, fitted[0], fitted[1])

    return X, Y, pdfX, pdfY, fitted, pdfXnorm, pdfYnorm, pdfYexpected


def u_nk_from_DF(data, temperature, eqTime, warnings=True):
    """
    Experimental. caveat emptor. Generate an alchemlyb-type u_nk dataframe from a dense dataframe generated by readFEPOUT
    Arguments: data[frame from readFEPOUT], temperature, eqTime (time to equilibrium), warnings (boolean, whether or not to print warnings)
    Returns: u_nk
    """
    from scipy.constants import R, calorie

    beta = 1 / (
        R / (1000 * calorie) * temperature
    )  # So that the final result is in kcal/mol
    u_nk = pd.pivot_table(
        data, index=["step", "FromLambda"], columns="ToLambda", values="dE"
    )
    # u_nk = u_nk.sort_index(level=0).sort_index(axis='columns') #sort the data so it can be interpreted by the BAR estimator
    u_nk = u_nk * beta
    u_nk.index.names = ["time", "fep-lambda"]
    u_nk.columns.names = [""]
    u_nk = u_nk.loc[u_nk.index.get_level_values("time") >= eqTime]

    # Shift and align values to be consistent with alchemlyb standards
    lambdas = list(
        set(u_nk.index.get_level_values(1)).union(set(u_nk.columns))
    )
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
            u_nk.loc[(slice(None), L), prev] = u_nk.loc[
                (slice(None), L), prev
            ].shift(1)
        except:
            if warnings:
                warns.add(L)

        prev = L

    if len(warns) > 0:
        print(f"Warning: lambdas={warns} not found in indices/columns")
    u_nk = u_nk.dropna(thresh=2)
    u_nk = u_nk.sort_index(level=1).sort_index(
        axis="columns"
    )  # sort the data so it can be interpreted by the BAR estimator
    return u_nk
