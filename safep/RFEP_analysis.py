from pathlib import Path
import re
import numpy as np
import pandas as pd

import safep


def main(logfile):
    logfile = Path(logfile)
    global_conf, _, biases, TI_traj = safep.parse_Colvars_log(logfile)

    restraint = get_changing_bias(biases)
    rest_name = restraint['name']
    cvs = restraint['colvar']
    print(f'Processing TI data for restraint {rest_name} on CVs {cvs}')
    path = logfile.parent  # We assume the colvars traj and log are in the same directory
    colvars_traj = get_colvars_traj_filename(global_conf, path)

    TI_cumulative, TI_per_window = get_cumulative_and_per_window_TI_data(restraint, colvars_traj)
    print_TI_summary(TI_cumulative)

    free_energy_gradients = get_precomputed_gradients(restraint, TI_traj, rest_name)
    make_and_save_TI_figure(TI_cumulative, TI_per_window, free_energy_gradients, logfile)


def get_precomputed_gradients(restraint, TI_traj, rest_name):
    free_energy_gradients = TI_traj[rest_name]['dAdL']
    lambdas = TI_traj[rest_name]['L']
    # if lambdaExponent >=2, set a zero derivative for lambda=0 (might be NaN in the data)
    if int(restraint['lambdaExponent']) >= 2 and np.isnan(free_energy_gradients[-1]):
        free_energy_gradients[-1] = 0.0
    if restraint['decoupling']:  # lambdas have opposite meaning
        # Convert to coupling
        free_energy_gradients = - np.array(free_energy_gradients)
        lambdas = 1.0 - np.array(lambdas)
    gradient_series = pd.Series(free_energy_gradients)
    gradient_series.index = lambdas
    return gradient_series


def get_colvar_column_names(colvars_traj):
    with open(colvars_traj, 'r', encoding="UTF8") as f:
        first_line = f.readline()
    columns = re.split(' +', first_line)[1:-1]
    return columns


def read_and_sanitize_TI_data(restraint, columns, colvars_traj):
    """
    trajectory could be read using colvartools (read multiple files etc)
    this might fail if vector variables are present
    """
    data_TI = pd.read_csv(colvars_traj, sep=r'\s+', names=columns, comment='#', index_col=0)
    # We could also take a user parameter to adjust this in post-processing, or do equilibration detection
    n_equil = restraint['targetEquilSteps']
    cv = restraint['colvar']
    data_TI = data_TI.rename(columns={cv: 'DBC'})
    schedule = restraint['lambdaSchedule']

    # Remove first samples of each window from analysis
    data_TI = data_TI[data_TI.index >= n_equil][1:]
    data_TI.index = data_TI.index - n_equil
    lambdas = np.minimum((data_TI.index.values - 1) // restraint['targetNumSteps'], len(schedule) - 1)
    data_lambdas = np.round([schedule[i] for i in lambdas], 3)
    data_TI.loc[:, 'L'] = data_lambdas
    data_TI = data_TI.iloc[1:]
    return data_TI


def print_TI_summary(TI_cumulative):
    free_energy = np.round(TI_cumulative['dG'][1], 1)
    error = np.round(TI_cumulative['error'][1], 1)
    print(f'ΔG_DBC = {free_energy} kcal/mol')
    print(f'Standard Deviation: {error} kcal/mol')


def get_colvars_traj_filename(global_conf, path):
    if 'traj_file' in global_conf:
        colvars_traj = path/global_conf['traj_file']
    else:
        colvars_traj = path/(global_conf['output_prefix']+'.colvars.traj')
    return colvars_traj


def get_changing_bias(biases):
    # Need to pick the right restraint if there are several
    # TODO look for harmonic wall with changing k
    for b in biases:
        if float(b['targetForceConstant']) >= 0 or b['decoupling'] in ['on', 'yes', 'true']:
            restraint = safep.make_harmonicWall_from_Colvars(b)
            break
    return restraint


def make_and_save_TI_figure(TI_cumulative, TI_per_window, free_energy_gradient, logfile):
    fig, axes = safep.plot_TI(TI_cumulative, TI_per_window, fontsize=20)
    axes[1].plot(free_energy_gradient.index, free_energy_gradient, marker='o', label='Colvars internal dA/dlambda', color='red')
    axes[1].legend()
    fig.savefig(Path(logfile).name.replace('.log', '_figures.png'))


def get_cumulative_and_per_window_TI_data(restraint, colvars_traj):
    columns = get_colvar_column_names(colvars_traj)
    data_TI = read_and_sanitize_TI_data(restraint, columns, colvars_traj)
    TI_per_window, TI_cumulative = safep.process_TI(data_TI, restraint, None)
    return TI_cumulative, TI_per_window


if __name__ == "__main__":
    main('RFEP_decouple.log')
