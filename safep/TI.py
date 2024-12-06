
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def process_TI(dataTI, restraint, Lsched):
    '''
    Arguments: the TI data, restraint, and lambda schedule
    Function: Calculate the free energy for each lambda value, aggregate the result, and estimate the error
    Returns: The free energies and associated errors as functions of lambda. Both per window and cumulative.
    '''
    dUs = {}
    for key, group in dataTI.groupby('L'):
        dUs[key] = [harmonicWall_dUdL(restraint, coord, key) for coord in group.DBC]

    # Note: here input parameter Lsched is ignored and overwritten
    Lsched = np.sort(list(dUs.keys()))
    dL = Lsched[1] - Lsched[0]
    TIperWindow = pd.DataFrame(index=Lsched)
    TIperWindow['dGdL'] = [np.mean(dUs[L])*dL for L in Lsched]
    TIperWindow['error'] = [np.std(dUs[L])*dL for L in Lsched] 

    TIcumulative = pd.DataFrame()
    TIcumulative['dG'] = np.cumsum(TIperWindow.dGdL)
    TIcumulative['error'] = np.sqrt(np.divide(np.cumsum(TIperWindow.error**2), np.arange(1,len(TIperWindow)+1)))
    
    return TIperWindow, TIcumulative


def plot_TI(cumulative, perWindow, width=8, height=4, PDFtype='KDE', hystLim=(-1,1), color='#0072B2', fontsize=12):
    fig, (cumAx,eachAx) = plt.subplots(2,1, sharex='col')

    # Cumulative change in kcal/mol
    cumAx.errorbar(cumulative.index, cumulative.dG, yerr=cumulative.error,marker=None, linewidth=1, color=color, label='Cumulative Change')
    finalEstimate = cumulative.dG[1]
    cumAx.axhline(finalEstimate, linestyle='-', color='gray', label=f'Final Value:\n{np.round(finalEstimate,1)}kcal/mol')
    cumAx.legend(fontsize=fontsize*0.75)                  
    cumAx.set_ylabel(r'Cumulative $\rm\Delta G_{\lambda}$'+'\n(kcal/mol)', fontsize=fontsize)

    # Per-window change in kcal/mol
    eachAx.errorbar(perWindow.index, perWindow.dGdL, marker=None, linewidth=1, yerr=perWindow.error, color=color)
    eachAx.set_ylabel(r'$\rm\Delta G_{\lambda}$'+'\n(kcal/mol)', fontsize=fontsize)

    eachAx.set_xlabel(r'$\lambda$', fontsize=fontsize)

    fig.set_figwidth(width)
    fig.set_figheight(height*3)
    fig.tight_layout()
    
    return fig, [cumAx,eachAx] 

def make_harmonicWall(FC=10, targetFC=0, targetFE=1, upperWalls=1, schedule=None, numSteps=1000, targetEQ=500, name='HW', lowerWalls=None, lambdaExp=1., decoupling=None):
    # Heuristic to set the default value of the decoupling parameter
    if decoupling is None:
        decoupling = (targetFC == 0 and FC != 0)

    HW = {'name':name, 'targetFC':targetFC, 'targetFE':targetFE, 'FC':FC, 'upperWalls':upperWalls, 'schedule':schedule,
           'numSteps':numSteps, 'targetEQ':targetEQ, 'lowerWalls':lowerWalls, 'lambdaExp':lambdaExp, 'decoupling':decoupling}

    return HW

def make_harmonicWall_from_Colvars(w):
    if (w['key'] != 'harmonicwalls'):
        k = w['key']
        print(f'Error: bias is not a harmonic wall (key: {k})')
        return None

    CVs = w['colvars'].strip("{}").replace(",", "").split()
    if len(CVs) != 1:
        print(f'Error: bias does not act on exactly one colvar (cvs: {CVs})')
        return None

    HW = {  'name': w['name'],
            'colvar': CVs[0],
            'FC': float(w['forceConstant']),
            'targetFC': float(w['targetForceConstant']),
            'lowerWalls': float(w['lowerWalls'].strip('{ }')),
            'upperWalls': float(w['upperWalls'].strip('{ }')),
            'numSteps': int(w['targetNumSteps']),
            'numStages': int(w['targetNumStages']),
            'targetEQ': int(w['targetEquilSteps']),
            'decoupling': w['decoupling'] in ['on', 'true', 'yes']} # interpret as Boolean
    # Deal with keywords that have changed name
    if 'targetForceExponent' in w:
        HW['targetFE'] = int(w['targetForceExponent'])
    if 'lambdaExponent' in w:
        HW['targetFE'] = int(w['lambdaExponent'])
    cleaned = w['lambdaSchedule'].strip("{}").replace(",", "")
    HW['schedule'] = [float(num) for num in cleaned.split()]
    return HW

def harmonicWall_U(HW, coord, L):
    d=0
    if HW['upperWalls'] and coord>HW['upperWalls']:
        d = coord-HW['upperWalls']
    elif HW['lowerWalls'] and coord<HW['lowerWalls']:
        d = coord-HW['lowerWalls']
    
    if d!=0:
        alpha = HW['targetFE']
        if HW['decoupling']:
            kL = (1.-L)**alpha * HW['FC']
        else:
            dk = HW['targetFC']-HW['FC']
            la = L**alpha
            kL = HW['FC']+la*dk

        U = 0.5*kL*(d**2)
    else:
        U=0
    return U

def harmonicWall_dUdL(HW, coord, L):
    d=0
    if HW['upperWalls'] and coord>HW['upperWalls']:
        d = coord-HW['upperWalls']
    elif HW['lowerWalls'] and coord<HW['lowerWalls']:
        d = coord-HW['lowerWalls']

    if d!=0:
        alpha = HW['targetFE']
        if HW['decoupling']:
            dkL = -alpha * (1.-L)**(alpha-1) * HW['FC']
        else:
            dk = HW['targetFC']-HW['FC']
            dla = alpha*L**(alpha-1)
            dkL = dla*dk

        dU = 0.5*dkL*(d**2)
    else:
        dU=0
    return dU
