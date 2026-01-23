import safep
from glob import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def main(logfile):
    global f, columns, dataTI
    # In[ ]:
    global_conf, colvars, biases, TI_traj = safep.parse_Colvars_log(logfile)
    # Need to pick the right restraint if there are several
    # TODO look for harmonic wall with changing k
    for b in biases:
        if float(b['targetForceConstant']) >= 0 or b['decoupling'] in ['on', 'yes', 'true']:
            DBC_rest = safep.make_harmonicWall_from_Colvars(b)
            break
    rest_name = DBC_rest['name']
    cvs = DBC_rest['colvar']
    print(f'Processing TI data for restraint {rest_name} on CVs {cvs}')
    path = os.path.dirname(logfile)  # We assume the colvars traj and log are in the same directory
    if 'traj_file' in global_conf:
        colvarsTraj = os.path.join(path, global_conf['traj_file'])
    else:
        colvarsTraj = os.path.join(path, global_conf['output_prefix'], '.colvars.traj')
    # In[ ]:
    # Extract precomputed TI info from traj file
    dAdL = TI_traj[rest_name]['dAdL']
    lambdas = TI_traj[rest_name]['L']
    # if lambdaExponent >=2, set a zero derivative for lambda=0 (might be NaN in the data)
    if int(DBC_rest['lambdaExponent']) >= 2 and np.isnan(dAdL[-1]):
        dAdL[-1] = 0.0
    if DBC_rest['decoupling']:  # lambdas have opposite meaning
        # Convert to coupling
        dAdL = - np.array(dAdL)
        lambdas = 1.0 - np.array(lambdas)
    # In[ ]:
    # Setup and processing of colvars data
    with open(colvarsTraj) as f:
        first_line = f.readline()
    columns = re.split(' +', first_line)[1:-1]
    # trajectory could be read using colvartools (read multiple files etc)
    # this might fail if vector variables are present
    dataTI = pd.read_csv(colvarsTraj, sep=r'\s+', names=columns, comment='#', index_col=0)
    # We could also take a user parameter to adjust this in post-processing, or do equilibration detection
    n_equil = DBC_rest['targetEquilSteps']
    cv = DBC_rest['colvar']
    dataTI = dataTI.rename(columns={cv: 'DBC'})
    schedule = DBC_rest['lambdaSchedule']
    dataTI = dataTI[dataTI.index >= n_equil][1:]  # Remove first samples of each window from analysis
    dataTI.index = dataTI.index - n_equil
    Ls = np.minimum((dataTI.index.values - 1) // DBC_rest['targetNumSteps'], len(schedule) - 1)
    dataLs = np.round([schedule[i] for i in Ls], 3)
    dataTI.loc[:, 'L'] = dataLs
    dataTI = dataTI.iloc[1:]
    TIperWindow, TIcumulative = safep.process_TI(dataTI, DBC_rest, None)
    # In[ ]:
    dG_DBC = np.round(TIcumulative['dG'][1], 1)
    error_DBC = np.round(TIcumulative['error'][1], 1)
    print(f'ΔG_DBC = {dG_DBC} kcal/mol')
    print(f'Standard Deviation: {error_DBC} kcal/mol')
    # In[ ]:
    ''' Plot the results. '''
    fig, axes = safep.plot_TI(TIcumulative, TIperWindow, fontsize=20)
    # This plot
    axes[1].plot(lambdas, np.array(dAdL), marker='o', label='Colvars internal dA/dlambda', color='red')
    axes[1].legend()
    plt.show()


if __name__ == "__main__":
    main('RFEP_decouple.log')




