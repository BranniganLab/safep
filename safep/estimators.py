import numpy as np
import pandas as pd
from alchemlyb.estimators import BAR


def get_exponential(u_nk):
    """
    Get exponential estimation of the change in free energy.
    Arguments: u_nk in alchemlyb format
    Returns:    l[ambdas],
                l_mid [lambda window midpoints],
                dG_f [forward estimates],
                dG_b [backward estimates]
    """

    groups = u_nk.groupby(level=1)
    dG = pd.DataFrame([])
    for name, group in groups:
        dG[name] = -np.log(np.mean(np.exp(-1 * group), axis=0))

    dG_f = np.diag(dG, k=-1)
    dG_b = np.diag(dG, k=1)

    l = dG.columns.to_list()
    l_mid = np.mean([l[1:], l[:-1]], axis=0)

    return l, l_mid, dG_f, dG_b


def get_BAR(bar):
    """
    Extract key information from an alchemlyb.BAR object. Useful for plotting.
    Arguments: a fitted BAR object
    Returns:    l[ambdas],
                l_mid [lambda window midpoints],
                f [the cumulative free energy],
                df [the per-window free energy],
                ddf [the per-window errors],
                errors [the cumulative error]
    """

    states = bar.states_

    f = bar.delta_f_.iloc[0, :]  # dataframe
    l = np.array([float(s) for s in states])
    # lambda midpoints for each window
    l_mid = 0.5 * (l[1:] + l[:-1])

    # FE differences are off diagonal
    df = np.diag(bar.delta_f_, k=1)

    # error estimates are off diagonal
    ddf = np.array([bar.d_delta_f_.iloc[i, i + 1] for i in range(len(states) - 1)])

    # Accumulate errors as sum of squares
    errors = np.array([np.sqrt((ddf[:i] ** 2).sum()) for i in range(len(states))])

    return l, l_mid, f, df, ddf, errors


def do_estimation(u_nk, method="both"):
    """
    Run both exponential and BAR estimators and return the results in tidy dataframes.
    Arguments: u_nk in the alchemlyb format, method of fitting (String: BAR, EXP, or both)
    Returns: per_window estimates (including errors), cumulative estimates (including errors)
    """
    u_nk = u_nk.sort_index(level=1)
    cumulative = pd.DataFrame()
    per_window = pd.DataFrame()
    if method in ["both", "BAR"]:
        bar = BAR()
        bar.fit(u_nk)
        ls, l_mids, fs, dfs, ddfs, errors = get_BAR(bar)

        bar_cumulative = pd.DataFrame()
        bar_per_window = pd.DataFrame()

        bar_cumulative[("BAR", "f")] = fs
        bar_cumulative[("BAR", "errors")] = errors
        bar_cumulative.index = ls

        bar_per_window[("BAR", "df")] = dfs
        bar_per_window[("BAR", "ddf")] = ddfs
        bar_per_window.index = l_mids

        cumulative = pd.concat([cumulative, bar_cumulative], axis=1)
        per_window = pd.concat([per_window, bar_per_window], axis=1)

    if method in ["both", "EXP"]:
        expl, expmid, dG_fs, dG_bs = get_exponential(u_nk)

        exp_cumulative = pd.DataFrame()
        exp_per_window = pd.DataFrame()

        exp_cumulative[("EXP", "ff")] = np.insert(np.cumsum(dG_fs), 0, 0)
        exp_cumulative[("EXP", "fb")] = np.insert(-np.cumsum(dG_bs), 0, 0)
        exp_cumulative.index = expl

        exp_per_window[("EXP", "dG_f")] = dG_fs
        exp_per_window[("EXP", "dG_b")] = dG_bs
        exp_per_window[("EXP", "difference")] = np.array(dG_fs) + np.array(dG_bs)
        exp_per_window.index = expmid

        cumulative = pd.concat([cumulative, exp_cumulative], axis=1)
        per_window = pd.concat([per_window, exp_per_window], axis=1)

    per_window.columns = pd.MultiIndex.from_tuples(per_window.columns)
    per_window = per_window.fillna(0)
    cumulative.columns = pd.MultiIndex.from_tuples(cumulative.columns)
    cumulative = cumulative.fillna(0)

    return per_window.copy(), cumulative.copy()


def get_dG_from_data(data, temperature):
    """Removed. I don't think this function is ever used. ES"""
    raise NotImplementedError("This function does not appear to be used anywhere." +
                              "If that is not the case, please contact the developer of safep.")
