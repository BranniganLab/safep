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


import numpy as np
import pandas as pd
import safep
import matplotlib.pyplot as plt
import re
from pymbar.timeseries import detect_equilibration
import argparse

import warnings #Suppress future warnings from pandas.
warnings.simplefilter(action='ignore', category=FutureWarning)

def subsample_TI_traj(dataTI, percent_min, percent_max):
    assert list(dataTI.columns).count('L'), "You must include lambda values for each non-zero energy sample in a column named 'L'"

    groups = dataTI.groupby('L')
    trimmed = []
    for key, grp in groups:
        size = len(grp)
        low = int(size*percent_min)
        hi = int(size*percent_max)
        toappend = grp.iloc[low:hi, :]
        trimmed.append(toappend)

    newData = pd.concat(trimmed)

    return newData

def read_colvar_traj(colvarsPath, DBC):
    with open(colvarsPath) as f:
        first_line = f.readline()
    columns = re.split(' +', first_line)[1:-1]
    dataTI = pd.read_csv(colvarsPath, delim_whitespace=True, names=columns, comment='#', index_col=0)
    dataTI = dataTI.rename(columns={'E_harmonicwalls2':'DBC_energy'})

    return dataTI


def do_TI_convergence(dataTI, DBC):
    forward = []
    ferr = []
    backward = []
    berr = []
    for x in np.linspace(0.9,0,10):
        subsampled = subsample_TI_traj(dataTI, x, 1)
        TIperWindow, TIcumulative = safep.process_TI(subsampled, DBC, DBC['schedule'])
        backward.append(TIcumulative.dG.iloc[-1])
        berr.append(TIcumulative.error.iloc[-1])

        subsampled = subsample_TI_traj(dataTI, 0, 1-x)
        TIperWindow, TIcumulative = safep.process_TI(subsampled, DBC, DBC['schedule'])
        forward.append(TIcumulative.dG.iloc[-1])
        ferr.append(TIcumulative.error.iloc[-1])

    return forward, ferr, backward, berr

def plot_TI_convergence(forward, ferr, backward, berr):
    fig, ax = plt.subplots()
    X = np.linspace(0.1,1,10)
    ax.plot(X, forward, label="forward-time subsampling")
    ax.fill_between(X, np.array(forward)+ferr, np.array(forward)-ferr, alpha = 0.5)
    ax.plot(X, backward, label="backward-time subsampling")
    ax.fill_between(X, np.array(backward)+berr, np.array(backward)-berr, alpha = 0.5)
    ax.set_xlabel("Fraction of non-zero samples")
    ax.set_ylabel("dG (kcal/mol)")
    fig.legend()
    fwd50 = forward[len(forward)//2-1]
    bwd50 = backward[len(backward)//2-1]
    delta50 = fwd50-bwd50
    avg50 = (fwd50+bwd50)/2
    txt = r"$\Delta G_{50} =$"+f"{np.round(delta50,2)}"
    ax.annotate(txt, (0.5, avg50))
    ax.plot([0.5, 0.5], [fwd50, bwd50], color="black")

    return fig, ax

def detect_equilibration_TI(dataTI, DBC):
    groups = dataTI.groupby('L')

    trimmed = []
    for key, grp in groups:
        start, _, eqsamples = detect_equilibration(grp.DBC, nskip=10)
        stride = int(len(grp)/eqsamples)
        toappend = grp.iloc[start::stride, :]
        diff = len(toappend)-eqsamples
        trimmed.append(toappend)

    newData = pd.concat(trimmed)
    newgroups = newData.groupby('L')

    lengths = {}
    for key, grp in newgroups:
        k = (key**DBC['targetFE'])*DBC['targetFC']

        lengths[k] = len(grp)

    return newData, lengths

def plot_TI_samples(lengths):
    fig, ax = plt.subplots()
    toplot = pd.Series(lengths)
    ax.plot(toplot)
    ax.invert_xaxis()
    ax.set_ylabel('Samples')
    ax.set_xlabel(r'$k_\lambda (\frac{kcal}{mol~\AA})$')

    return fig, ax

def parse_TI_log(logpath):
    with open(logpath) as logfile:
        wholelog = logfile.read()
        loglines = wholelog.split('\n')
        restraintREGEX = 'harmonicwalls.*initialize'
        restraint = re.search(restraintREGEX, wholelog, re.DOTALL)
        restraintStr = restraint.group(0)
        
        restraint_name = re.search('(.*\")(.*)(\".*\n.*\{\ DBC\ \})', restraintStr).group(2)
        steps_per_lambda = int(re.search('(targetNumSteps\ *=\ *)(\d+)', restraintStr).group(2))
        Lsched = np.float64(re.search('(.*lambdaSchedule\ =\ \{)(.*)(\})', restraintStr).group(2).split(", "))
        FC = float(re.search(f'({restraint_name}.*forceConstant\ *=\ *)(\d+)', restraintStr, re.DOTALL).group(2))
        targetFC = float(re.search('(targetForceConstant\ *=\ *)(\d+)', restraintStr).group(2))
        exponent = float(re.search('(targetForceExponent\ *=\ *)(\d+)', restraintStr).group(2))

        widthRegex = f'({restraint_name}'+'.*upperWalls\ *=\ *{\ )(\d+)(\ })'
        DBCwidth = float(re.search(widthRegex, restraintStr, re.DOTALL).group(2))
        
        eqstepsRegex = f'({restraint_name}'+'.*targetEquilSteps\ *=\ *)(\d+)'
        eqsteps = int(re.search(eqstepsRegex, restraintStr, re.DOTALL).group(2))

    DBC = safep.make_harmonicWall(FC=FC, 
        targetFC=targetFC, 
        targetFE=exponent, 
        upperWalls=DBCwidth, 
        targetEQ=eqsteps, 
        numSteps=steps_per_lambda, 
        name=restraint, 
        schedule=Lsched)

    return DBC, Lsched

def l_from_energy(dataTI, DBC):
    guessedLs = [
        safep.guessL(
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

    return dataTI

def l_from_sched(dataTI, Lsched, DBC):
    Ls = (dataTI.index.values-1)//DBC['numSteps']

    dataLs = np.round([DBC['schedule'][i] for i in Ls], 3)
    dataTI.loc[:,'L'] = dataLs
    dataTI = dataTI.iloc[1:]

    return dataTI

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        prog = "processTI",
        description = "Process the outputs from a NAMD TI calculation."
        )
    parser.add_argument('namdlog')
    parser.add_argument('cvconfig')
    parser.add_argument('traj_path')
    parser.add_argument('-o', '--outputDirectory', type=str, default='.')
    parser.add_argument('-d', '--detectEquilibrium', action='store_true')

    args = parser.parse_args()

    print("Reading...")
    DBC, Lsched = setup_TI_analysis(args.namdlog)
    dataTI = read_colvar_traj(args.traj_path, DBC)
    if 'DBC_energy' in dataTI.columns:
        dataTI = l_from_energy(dataTI)
    else:
        dataTI = l_from_sched(dataTI, Lsched)
    
    print("Processing...")
    forward, ferr, backward, berr = do_TI_convergence(dataTI, DBC)
    TIperWindow, TIcumulative = safep.process_TI(dataTI, DBC, Lsched)

    print("Plotting...")
    fig, ax = plot_convergence(forward, ferr, backward, berr)
    plt.savefig(f'{args.outputDirectory}/TI_convergence.pdf')

    fig, ax = safep.plot_TI(TIcumulative, TIperWindow, fontsize=20)
    plt.savefig(f'{args.outputDirectory}/TI_general.pdf')

    if args.detectEquilibrium:
        newData, lengths = detect_equilibration_TI(dataTI, DBC)
        TIperWindow, TIcumulative = safep.process_TI_DBC(newData, DBC)
        fig, ax = plot_samples(lengths)
        plt.savefig(f'{args.outputDirectory}/decorrelated_TI_samples.pdf')

    print("Done")


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
