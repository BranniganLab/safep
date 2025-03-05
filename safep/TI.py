
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.integrate


def process_TI(dataTI, restraint, Lsched):
    '''
    Arguments: the TI data, restraint, and lambda schedule
    Function: Calculate the free energy for each lambda value, aggregate the result, and estimate the error
    Returns: The free energies and associated errors as functions of lambda. Both per window and cumulative.
    '''
    dUs = {}
    # Iterate over lambdas
    for key, group in dataTI.groupby('L'):
        dUs[key] = harmonicWall_dUdL(restraint, group.DBC, key)

    Lsched = np.sort(list(dUs.keys())) # FIXME parameter Lsched is ignored and overwritten
    dL = Lsched[1] - Lsched[0]         # FIXME assumes that the lambda schedule is uniformly spaced
    TIperWindow = pd.DataFrame(index=Lsched)
    TIperWindow['dGdL'] = [np.mean(dUs[L]) for L in Lsched]
    TIperWindow['error'] = [np.std(dUs[L]) for L in Lsched]

    if Lsched[-1] < 1.0:
        # TODO test lambdaExponent >= 2
        lastPoint = pd.DataFrame({
            "dGdL": pd.Series([0.0], index=[1.0]),
            'error': pd.Series([0.0], index=[1.0])
        })
        TIperWindow = pd.concat([TIperWindow, lastPoint])
        Lsched = np.concatenate([Lsched, [1.0]])

    TIcumulative = pd.DataFrame()
    TIcumulative['dG'] = scipy.integrate.cumulative_trapezoid(TIperWindow.dGdL, x=Lsched, initial=0)
    TIcumulative.set_index(Lsched, inplace=True)

    # Estimate square error for trapezoid rule by averaging errors in neighboring bins
    sq_error = np.array(TIperWindow.error**2)
    sq_error = 0.5 * (sq_error[1:] + sq_error[:-1])
    error = np.sqrt(np.cumsum(sq_error * dL**2))
    TIcumulative['error'] = np.concatenate([[0.0], error]) # Include initial 0 error

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
    '''
    This method should rarely be used, as long as a log file exists that can be parsed by parse_Colvars_log
    and the result fed to make_harmonicWall_from_Colvars
    '''

    # Heuristic to set the default value of the decoupling parameter
    if decoupling is None:
        decoupling = (targetFC == 0 and FC != 0)

    HW = {'name':name, 'targetForceConstant':targetFC, 'lambdaExponent':targetFE, 'forceConstant':FC, 'upperWalls':upperWalls, 'schedule':schedule,
           'targetNumSteps':numSteps, 'targetEquilSteps':targetEQ, 'lowerWalls':lowerWalls, 'lambdaExp':lambdaExp, 'decoupling':decoupling}

    return HW

def make_harmonicWall_from_Colvars(restraint_conf):
    '''
    Input: a restraint configuration as a dict produced by parse_Colvars_log
    '''

    if (restraint_conf['key'] != 'harmonicwalls'):
        keyword = restraint_conf['key']
        print(f'Error: bias is not a harmonic wall (keyword: {keyword})')
        return None

    CVs = restraint_conf['colvars'].strip("{}").replace(",", "").split()
    if len(CVs) != 1:
        raise RuntimeError (f'Error: bias does not act on exactly one colvar (cvs: {CVs})')

    HW = {  'name': restraint_conf['name'],
            'colvar': CVs[0],
            'forceConstant': float(restraint_conf['forceConstant']),
            'targetForceConstant': float(restraint_conf['targetForceConstant']),
            'lowerWalls': float(restraint_conf['lowerWalls'].strip('{ }')),
            'upperWalls': float(restraint_conf['upperWalls'].strip('{ }')),
            'targetNumSteps': int(restraint_conf['targetNumSteps']),
            'targetNumStages': int(restraint_conf['targetNumStages']),
            'targetEquilSteps': int(restraint_conf['targetEquilSteps']),
            'decoupling': restraint_conf['decoupling'] in ['on', 'true', 'yes']} # interpret as Boolean
    # Support legacy keyword
    if 'targetForceExponent' in restraint_conf:
        HW['lambdaExponent'] = int(restraint_conf['targetForceExponent'])
    if 'lambdaExponent' in restraint_conf:
        HW['lambdaExponent'] = int(restraint_conf['lambdaExponent'])
    cleaned = restraint_conf['lambdaSchedule'].strip("{}").replace(",", "")
    if len(cleaned.split()) > 0:
        HW['lambdaSchedule'] = [float(num) for num in cleaned.split()]
    else:
        HW['lambdaSchedule'] = np.linspace(0, 1, HW['targetNumStages'] + 1)
    return HW

def harmonicWall_U(HW, coord, L):
    d=0
    if HW['upperWalls'] and coord>HW['upperWalls']:
        d = coord-HW['upperWalls']
    elif HW['lowerWalls'] and coord<HW['lowerWalls']:
        d = coord-HW['lowerWalls']
    
    if d!=0:
        alpha = HW['lambdaExponent']
        if HW['decoupling']:
            kL = (1.-L)**alpha * HW['forceConstant']
        else:
            dk = HW['targetForceConstant']-HW['forceConstant']
            la = L**alpha
            kL = HW['forceConstant']+la*dk

        U = 0.5*kL*(d**2)
    else:
        U=0
    return U


# Calculate dUdL but using matrix primitives
def harmonicWall_dUdL(HW, coords, L):
    d = np.zeros_like(coords)
    if HW['upperWalls']:
        keep_mask = coords > HW['upperWalls']
        d[keep_mask] = (coords - HW['upperWalls'])[keep_mask]
    elif HW['lowerWalls']:
        keep_mask = coords < HW['lowerWalls']
        d[keep_mask] = (coords - HW['lowerWalls'])[keep_mask]
    
    dk = HW['targetFC'] - HW['FC']
    dla = HW['targetFE']*L**(HW['targetFE']-1)
    kL = HW['FC']+ dla*dk
    # In the old, serial version of this code we checked whether d was 0.
    # It doesn't actually matter since pow(d, 2) == 0 so we don't bother with an elementwise check.
    return 0.5*kL*np.pow(d, 2)


def harmonicWall_dUdL_serial(HW, coord, L):
    d=0
    if HW['upperWalls'] and coord>HW['upperWalls']:
        d = coord-HW['upperWalls']
    elif HW['lowerWalls'] and coord<HW['lowerWalls']:
        d = coord-HW['lowerWalls']

    if d!=0:
        alpha = HW['lambdaExponent']
        if HW['decoupling']:
            dkL = -alpha * (1.-L)**(alpha-1) * HW['forceConstant']
        else:
            dk = HW['targetForceConstant']-HW['forceConstant']
            dla = alpha*L**(alpha-1)
            dkL = dla*dk

        dU = 0.5*dkL*(d**2)
    else:
        dU=0
    return dU
