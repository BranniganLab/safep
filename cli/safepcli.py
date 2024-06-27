#!/usr/bin/env python3
import argparse
import configparser
import glob
import re
from collections import defaultdict
from multiprocessing import Pool
import pandas as pd
import scipy
import numpy as np
import safep

"""
; Sample configuration file
; DEFAULT must be capitalized for some reason

[DEFAULT]
temperature=303.15
figout_path=/tmp

[calc1]
bound=/nas1/foo/bar/whatever/namd/disprodA*.fepout
rfep=/nas1/foo/bar/whatever/namd/RFEP.colvars.traj
desolv=/nas1/bar/baz/ligdisA*.fepout
dbcwidth=5
comradius=6

[calc2]
bound=/nas1/foo/bar/whatever/namd/disprodA*.fepout
rfep=/nas1/foo/bar/whatever/namd/RFEP.colvars.traj
desolv=/nas1/bar/baz/ligdisA*.fepout
dbcwidth=4
comradius=5

"""
def main():
    ap = argparse.ArgumentParser(description='SAFEP command line interface')
    ap.add_argument('config', help='Configuration file')
    args = ap.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    temperature = float(config['DEFAULT']['temperature'])
    R = scipy.constants.R/(1000*scipy.constants.calorie) # gas constant in kcal/(mol K)
    global_params = {}
    global_params['temperature'] = temperature
    global_params['RT'] = R * temperature # RT in kcal/mol

    # We will run each FEP estimation in a separate process, managed by multiprocessing.Pool, which will=
    # automatically limit the number of concurrent processes to the number of CPU cores it detects on the
    # machine by default.

    # The worker parameters array looks like:
    # [(params: dict, fepouts: list|str), ...]
    worker_params = []
    # Stores the results of the bulk calculations
    bulk_calc_cache = {}
    # Keeps track of which bulk calculations are the same
    bulk_pointers = {}
    # Each overall calculation, including bound, bulk, volumetric, DBC TI, is identified by a calc_name
    # and a leg_name. The leg_name would be either 'bound' or 'bulk'. These strings are used below when
    # we are gathering the results.
    for calc_name in config.sections():
        params = global_params.copy()
        params['calc_name'] = calc_name
        params['detectEQ'] = True
        params['decorrelate'] = True

        # Set up bound leg calculation
        params['leg_name'] = 'bound'
        worker_params.append((params.copy(), config[calc_name]['bound']))

        # Set up bulk leg calculation
        params['leg_name'] = 'bulk'
        # ...but only actually set up the calculation if we haven't already.
        # Otherwise, leave a pointer to the calc_name that contained the bulk calculation we'll use
        if str(config[calc_name]['bulk']) not in bulk_calc_cache:
            worker_params.append((params.copy(), config[calc_name]['bulk']))
            bulk_calc_cache[str(config[calc_name]['bulk'])] = calc_name
        else:
            prev_calc_name = bulk_calc_cache[str(config[calc_name]['bulk'])]
            bulk_pointers[calc_name] = prev_calc_name

    # Actually run the FEP estimations in parallel. The Pool.map function returns an array of results
    # that were returned by the individual invocations of fep_worker().
    with Pool() as p:
        worker_results = p.map(fep_worker, worker_params)

    # Gather and organize results, according to calc_name and leg_name for each
    final_results = defaultdict(dict)
    for i in range(len(worker_results)):
        params_i, _ = worker_params[i]
        cumulative, per_window, u_nk = worker_results[i]
        calc_name, leg_name = params_i['calc_name'], params_i['leg_name']
        final_results[calc_name][f'{leg_name} cumulative'] = cumulative
        final_results[calc_name][f'{leg_name} per_window'] = per_window
        final_results[calc_name][f'{leg_name} u_nk'] = u_nk

    # TODO: Make all the pretty figures
    for calc_name in config.sections():
        # Look in our bulk calculation cache as necessary. We stored the e
        if 'bulk cumulative' not in final_results[calc_name]:
            cumulative_bulk = final_results[bulk_pointers[calc_name]]['bulk cumulative']
        else:
            cumulative_bulk = final_results[calc_name]['bulk cumulative']
        RT = global_params['RT']
        dG_bulk = np.round(cumulative_bulk.BAR.f.iloc[-1] * RT, 1)
        error_bulk = np.round(cumulative_bulk.BAR.errors.iloc[-1] * RT, 1)

        # Volumetric restraint contribution
        molar = 1660  # cubic angstroms per particle in a one molar solution
        dG_V = np.round(-float(RT) * np.log(4 / 3 * scipy.pi * float(config[calc_name]['comradius']) ** 3 / molar), 1)
        error_V = 0.0

        # Process the DBC TI RFEP for each calc_name. We don't use worker processes for this because it's typically
        # a relatively small amount of computation, but there's no fundamental reason not to if we choose.
        ti_per_window, ti_cumulative = process_dbc_ti(calc_name, config[calc_name])
        dG_DBC = np.round(ti_cumulative['dG'][1], 1)
        error_DBC = np.round(ti_cumulative['error'][1], 1)

        cumulative = final_results[calc_name]['bound cumulative']
        dG_site = np.round(cumulative.BAR.f.iloc[-1] * RT, 1)
        error_site = np.round(cumulative.BAR.errors.iloc[-1] * RT, 1)

        dG_binding = dG_bulk + dG_V + dG_DBC - dG_site
        error_binding = np.sqrt(np.sum(np.array([error_bulk, error_V, error_site, error_DBC]) ** 2))
        print(calc_name, f'dG_bulk = {dG_bulk:.2f} (err: {error_bulk:.2f} kcal/mol)')
        print(calc_name, f'dG_V = {dG_V:.2f} (err: {error_V:.2f} kcal/mol)')
        print(calc_name, f'dG_DBC = {dG_DBC:.2f} (err: {error_DBC:.2f} kcal/mol)')
        print(calc_name, f'dG_site = {dG_site:.2f} (err: {error_site:.2f} kcal/mol)')
        print(calc_name, f'dG_binding = {dG_binding:.2f} (err: {error_binding:.2f} kcal/mol)')


def process_dbc_ti(calc_name, params):
    """Process the DBC TI output and return per-window and cumulative energy curves."""
    with open(params['rfep']) as f:
        first_line = f.readline()
    columns = re.split(' +', first_line)[1:-1]
    data_ti = pd.read_csv(params['rfep'], delim_whitespace=True, names=columns, comment='#', index_col=0)
    data_ti= data_ti[data_ti.index >= 1000][1:]
    data_ti.index = data_ti.index - 1000

    n_lambdas = int(params['rfep_n'])

    dist = safep.make_harmonicWall(FC=float(params['rfep_fc']),
                                   upperWalls=float(params['comradius']),
                                   name='distance_restraint')
    # TODO: Lots of hardcoded parameters!
    lambda_schedule = np.linspace(1, 0, n_lambdas)
    dbc = safep.make_harmonicWall(FC=0,
                                  targetFC=float(params['rfep_fc']),
                                  targetFE=6,
                                  upperWalls=float(params['dbcwidth']),
                                  targetEQ=500,
                                  numSteps=300000,
                                  name='harmonicwalls2',
                                  schedule=lambda_schedule)
    Ls = (data_ti.index.values - 1) // dbc['numSteps']
    Ls[0] = 0
    Ls[Ls == n_lambdas] = n_lambdas - 1  # This is a small hack in case there are extra samples for the last window

    data_ls = np.round([dbc['schedule'][i] for i in Ls], 3)
    data_ti.loc[:, 'L'] = data_ls
    data_ti = data_ti.iloc[1:]

    ti_per_window, ti_cumulative = safep.process_TI(data_ti, dbc, lambda_schedule)
    return ti_per_window, ti_cumulative


def fep_worker(args):
    """Worker function, invoked by Pool, to do a single FEP estimation."""
    params, fepouts = args
    fepouts = glob_as_necessary(fepouts)
    u_nk = safep.read_and_process(fepouts, params['temperature'], decorrelate=params['decorrelate'],
                                   detectEQ=params['detectEQ'])
    per_window, cumulative = safep.do_estimation(u_nk)
    return cumulative, per_window, u_nk


def glob_as_necessary(path):
    """Takes a single glob expression or a list of glob expressions, and
    globs them into a list of files. Prefixing with '@' specifies a filename
    containing these things.
    """

    # A string could be either a reference to a filename containing paths (if it starts with '@')
    # or a glob expression. A list is presumed to be a list of glob expressions.
    # I guess we don't support a list of references to filenames, which I suppose is a disadvantage,
    # but on the other hand this makes it impossible for a file to refer to another file and
    # recurse to infinity.
    if type(path) == str:
        if path.startswith('@'):
            with open(path[1:], 'r') as f:
                return glob_as_necessary(f.readlines())
        else:
            return glob.glob(path)
    elif type(path) == list:
        files = []
        for filename in path:
            files.extend(glob.glob(filename.strip()))
        return files
    else:
        return []


if __name__ == '__main__':
    main()
