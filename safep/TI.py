import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def guessL(U, k0, k1, x, xwall, alpha, debug=False):
    """
    Designed to simplify DBC TI calculations. Not guaranteed for other purposes.
    Estimates a lambda value based on the energy, colvar value, and parameters of a restraint.
    If the restraint is not active, return nan.

    Parameters
    ----------
    U : float
        Instantaneous potential energy of the restraint.
    k0 : float
        The force constant at L=0.
    k1 : float
        The force constant at L=1.
    x : float
        The instantaneous value of the collective variable.
    xwall : float
        The upper wall of the collective variable.
    alpha : float
        The "force exponent" that softens the lambda schedule near L=0.
    debug : bool, optional
        If true, print debugging information. The default is False.

    Returns
    -------
    L : float
        The estimated lambda value.

    """
    dx = x - xwall
    if dx > 0:
        dx2 = dx**2
        dk = k1 - k0
        first = (2 * U) / (dk * dx2)
        second = k0 / dk
        exp = 1 / alpha
        La = first - second
        L = La**exp
    else:
        L = np.nan

    if debug:
        print(
            f"dx={dx}\ndx2={dx2}\ndk={dk}\nfirst={first}\nsecond={second}\nexp={exp}\nLa={La}\nL={L}"
        )

    return L


def process_TI_DBC(dataTI, DBC):
    """
    Special case of TI processing that uses the energies to estimate lambda
    instead of using information about the expected lambda schedule which has
    been found to be more error-prone.

    Parameters
    ----------
    dataTI : pandas dataframe
        Basic dataframe containing information from a TI calculation.
    DBC : TYPE
        DESCRIPTION.

    Returns
    -------
    TIperWindow : TYPE
        DESCRIPTION.
    TIcumulative : TYPE
        DESCRIPTION.

    """

    guessedLs = [
        guessL(
            U,
            DBC["FC"],
            DBC["targetFC"],
            dbc,
            DBC["upperWalls"],
            DBC["targetFE"],
        )
        for U, dbc in zip(dataTI.DBC_energy, dataTI.DBC)
    ]
    dataTI["L"] = np.round(guessedLs, 6)

    dUs = {}
    for key, group in dataTI.groupby("L"):
        dUs[key] = [harmonicWall_dUdL(DBC, coord, key) for coord in group.DBC]

    Lsched = np.sort(list(dUs.keys()))
    dL = Lsched[1] - Lsched[0]
    TIperWindow = pd.DataFrame(index=Lsched)
    TIperWindow["dGdL"] = [np.mean(dUs[L]) * dL for L in Lsched]
    TIperWindow["error"] = [np.std(dUs[L]) * dL for L in Lsched]

    TIcumulative = pd.DataFrame()
    TIcumulative["dG"] = np.cumsum(TIperWindow.dGdL)
    TIcumulative["error"] = np.sqrt(
        np.divide(
            np.cumsum(TIperWindow.error**2),
            np.arange(1, len(TIperWindow) + 1),
        )
    )

    return TIperWindow, TIcumulative


def process_TI(dataTI, restraint, Lsched):
    """
    Arguments: the TI data, restraint, and lambda schedule
    Function: Calculate the free energy for each lambda value, aggregate the result, and estimate the error
    Returns: The free energies and associated errors as functions of lambda. Both per window and cumulative.
    """
    dUs = {}
    for key, group in dataTI.groupby("L"):
        dUs[key] = [
            harmonicWall_dUdL(restraint, coord, key) for coord in group.DBC
        ]

    Lsched = np.sort(list(dUs.keys()))
    dL = Lsched[1] - Lsched[0]
    TIperWindow = pd.DataFrame(index=Lsched)
    TIperWindow["dGdL"] = [np.mean(dUs[L]) * dL for L in Lsched]
    TIperWindow["error"] = [np.std(dUs[L]) * dL for L in Lsched]

    TIcumulative = pd.DataFrame()
    TIcumulative["dG"] = np.cumsum(TIperWindow.dGdL)
    TIcumulative["error"] = np.sqrt(
        np.divide(
            np.cumsum(TIperWindow.error**2),
            np.arange(1, len(TIperWindow) + 1),
        )
    )

    return TIperWindow, TIcumulative


def plot_TI(
    cumulative,
    perWindow,
    width=8,
    height=4,
    PDFtype="KDE",
    hystLim=(-1, 1),
    color="#0072B2",
    fontsize=12,
):
    fig, (cumAx, eachAx) = plt.subplots(2, 1, sharex="col")

    # Cumulative change in kcal/mol
    cumAx.errorbar(
        cumulative.index,
        cumulative.dG,
        yerr=cumulative.error,
        marker=None,
        linewidth=1,
        color=color,
        label="Cumulative Change",
    )
    finalEstimate = cumulative.dG[1]
    cumAx.axhline(
        finalEstimate,
        linestyle="-",
        color="gray",
        label=f"Final Value:\n{np.round(finalEstimate,1)}kcal/mol",
    )
    cumAx.legend(fontsize=fontsize * 0.75)
    cumAx.set_ylabel(
        r"Cumulative $\rm\Delta G_{\lambda}$" + "\n(kcal/mol)",
        fontsize=fontsize,
    )

    # Per-window change in kcal/mol
    eachAx.errorbar(
        perWindow.index,
        perWindow.dGdL,
        marker=None,
        linewidth=1,
        yerr=perWindow.error,
        color=color,
    )
    eachAx.set_ylabel(
        r"$\rm\Delta G_{\lambda}$" + "\n(kcal/mol)", fontsize=fontsize
    )

    eachAx.set_xlabel(r"$\lambda$", fontsize=fontsize)

    fig.set_figwidth(width)
    fig.set_figheight(height * 3)
    fig.tight_layout()

    return fig, [cumAx, eachAx]


def make_harmonicWall(
    FC=10,
    targetFC=0,
    targetFE=1,
    upperWalls=1,
    schedule=None,
    numSteps=1000,
    targetEQ=500,
    name="HW",
    lowerWalls=None,
):
    HW = {
        "name": name,
        "targetFC": targetFC,
        "targetFE": targetFE,
        "FC": FC,
        "upperWalls": upperWalls,
        "schedule": schedule,
        "numSteps": numSteps,
        "targetEQ": targetEQ,
        "lowerWalls": lowerWalls,
    }
    return HW


def harmonicWall_U(HW, coord, L):
    d = 0
    if HW["upperWalls"] and coord > HW["upperWalls"]:
        d = coord - HW["upperWalls"]
    elif HW["lowerWalls"] and coord < HW["lowerWalls"]:
        d = coord - HW["lowerWalls"]

    if d != 0:
        dk = HW["targetFC"] - HW["FC"]
        la = L ** HW["targetFE"]
        kL = HW["FC"] + la * dk
        U = 0.5 * kL * (d**2)
    else:
        U = 0
    return U


def harmonicWall_dUdL(HW, coord, L):
    d = 0
    if HW["upperWalls"] and coord > HW["upperWalls"]:
        d = coord - HW["upperWalls"]
    elif HW["lowerWalls"] and coord < HW["lowerWalls"]:
        d = coord - HW["lowerWalls"]

    if d != 0:
        dk = HW["targetFC"] - HW["FC"]
        dla = HW["targetFE"] * L ** (HW["targetFE"] - 1)
        kL = HW["FC"] + dla * dk
        dU = 0.5 * kL * (d**2)
    else:
        dU = 0
    return dU


# Probably not relevant for the tutorial

# if np.isnan(dataTI.E_dist.iloc[0]):
#    dataTI.loc[:,'E_dist'] = [HW_U(Dist, coord, 0) for coord in dataTI.distance]
# if np.isnan(dataTI.E_DBC.iloc[0]):
#    dataTI.loc[:,'E_DBC'] = [HW_U(DBC, coord, L) for coord, L in zip(dataTI.DBC, dataTI.L)]

# plt.plot(dataTI.E_dist, label='spherical restraint', alpha=0.5)
# plt.plot(dataTI.E_DBC, label='DBC restraint', alpha=0.5)
# plt.legend()
# plt.savefig(f'{path}_restraint_overlap.pdf')

# plt.plot(RFEPdat, label='colvars estimate', color='red')
# plt.errorbar(Lsched, TIperWindow['dGdL'], yerr=TIperWindow['error'], label='Notebook Estimate', linestyle='-.', color='black')
# plt.savefig(f'{path}_TI_vs_colvarEst.pdf')
# plt.legend()
