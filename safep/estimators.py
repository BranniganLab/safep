"""
Estimators for getting free energies from energy differences.
"""

import numpy as np
import pandas as pd
from alchemlyb.estimators import BAR

# from .helpers import


def get_exponential(u_nk):
    """
    Get exponential estimation of the change in free energy.

    Args:
        u_nk in alchemlyb format

    Returns:
        l[ambdas]
        l_mid [lambda window midpoints]
        delta_free_energy_f [forward estimates]
        delta_free_energy_b [backward estimates]
    """

    groups = u_nk.groupby(level=1)
    delta_free_energy = pd.DataFrame([])
    for name, group in groups:
        delta_free_energy[name] = np.log(np.mean(np.exp(-1 * group), axis=0))

    delta_free_energy_f = np.diag(delta_free_energy, k=1)
    delta_free_energy_b = np.diag(delta_free_energy, k=-1)

    l = delta_free_energy.columns.to_list()
    l_mid = np.mean([l[1:], l[:-1]], axis=0)

    return l, l_mid, delta_free_energy_f, delta_free_energy_b


def get_BAR(bar_estimate):
    """
    Extract key information from an alchemlyb.BAR object. Useful for plotting.

    Args:
        a fitted BAR object

    Returns:
        l[ambdas]
        l_mid [lambda window midpoints]
        f [the cumulative free energy]
        df [the per-window free energy]
        ddf [the per-window errors]
        errors [the cumulative error]
    """

    states = bar_estimate.states_

    f = bar_estimate.delta_f_.iloc[0, :]  # dataframe
    l = np.array([float(s) for s in states])
    # lambda midpoints for each window
    l_mid = 0.5 * (l[1:] + l[:-1])

    # FE differences are off diagonal
    df = np.diag(bar_estimate.delta_f_, k=1)

    # error estimates are off diagonal
    ddf = np.array(
        [bar_estimate.d_delta_f_.iloc[i, i + 1] for i in range(len(states) - 1)]
    )

    # Accumulate errors as sum of squares
    errors = np.array([np.sqrt((ddf[:i] ** 2).sum()) for i in range(len(states))])

    return l, l_mid, f, df, ddf, errors


def do_estimation(u_nk, method="both"):
    """
    Run both exponential and BAR estimators and return the results in tidy dataframes.

    Args:
        u_nk: in the alchemlyb format
        method: of fitting (String: BAR, EXP, or both)

    Returns:
        perWindow estimates (including errors)
        cumulative estimates (including errors)
    """
    u_nk = u_nk.sort_index(level=1)
    cumulative = pd.DataFrame()
    per_window = pd.DataFrame()
    if method == "both" or method == "BAR":
        bar_estimate = BAR()
        bar_estimate.fit(u_nk)
        ls, l_mids, fs, dfs, ddfs, errors = get_BAR(bar_estimate)

        cumulative[("BAR", "f")] = fs
        cumulative[("BAR", "errors")] = errors
        cumulative.index = ls

        per_window[("BAR", "df")] = dfs
        per_window[("BAR", "ddf")] = ddfs
        per_window.index = l_mids

    if method == "both" or method == "EXP":
        expl, expmid, forward_fe_change, backward_fe_change = get_exponential(u_nk)

        cumulative[("EXP", "ff")] = np.insert(np.cumsum(forward_fe_change), 0, 0)
        cumulative[("EXP", "fb")] = np.insert(-np.cumsum(backward_fe_change), 0, 0)
        cumulative.index = expl

        per_window[("EXP", "delta_free_energy_f")] = forward_fe_change
        per_window[("EXP", "delta_free_energy_b")] = backward_fe_change
        per_window[("EXP", "difference")] = np.array(forward_fe_change) + np.array(
            backward_fe_change
        )
        per_window.index = expmid

    per_window.columns = pd.MultiIndex.from_tuples(per_window.columns)
    per_window = per_window.fillna(0)
    cumulative.columns = pd.MultiIndex.from_tuples(cumulative.columns)
    cumulative = cumulative.fillna(0)

    return per_window.copy(), cumulative.copy()


# Light-weight exponential estimator. Requires alternative parser.
def get_delta_free_energy_from_data(data, temperature):
    """
    Extract the forward and backward delta G's from an alchemlyb u_nk dataframe.
    Probably unnecessary.

    Args:
        data (pd.DataFrame): an alchemlyb-formatted pandas dataframe
        temperature: the temperature of the simulations

    Returns:
        The forward delta_free_energy list
        The backward delta_free_energy list
    """
    from scipy.constants import R, calorie

    beta = 1 / (
        R / (1000 * calorie) * temperature
    )  # So that the final result is in kcal/mol

    groups = data.groupby(level=0)
    delta_free_energy = []
    for name, group in groups:
        up_mask = group.up
        delta_energy = group.dE
        to_append = [
            name,
            -1 * np.log(np.mean(np.exp(-beta * delta_energy[up_mask]))),
            1,
        ]
        delta_free_energy.append(to_append)
        to_append = [
            name,
            -1 * np.log(np.mean(np.exp(-beta * delta_energy[~up_mask]))),
            0,
        ]
        delta_free_energy.append(to_append)

    delta_free_energy = pd.DataFrame(
        delta_free_energy, columns=["window", "delta_free_energy", "up"]
    )
    delta_free_energy = delta_free_energy.set_index("window")

    delta_free_energy_f = delta_free_energy.loc[delta_free_energy.up == 1]
    delta_free_energy_b = delta_free_energy.loc[delta_free_energy.up == 0]

    delta_free_energy_f = delta_free_energy_f.delta_free_energy.dropna()
    delta_free_energy_b = delta_free_energy_b.delta_free_energy.dropna()

    return delta_free_energy_f, delta_free_energy_b
