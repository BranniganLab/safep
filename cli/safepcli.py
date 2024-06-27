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

    cp = configparser.ConfigParser()
    cp.read(args.config)

    temperature = float(cp['DEFAULT']['temperature'])
    R = scipy.constants.R/(1000*scipy.constants.calorie) # gas constant in kcal/(mol K)
    global_params = {}
    global_params['temperature'] = temperature
    global_params['RT'] = R * temperature # RT in kcal/mol

    # Perhaps it's worth pulling out all the individual read_and_process and do_estimation runs
    # and putting those all in the pool, then redistributing them for whatever further processing

    # Worker parameters looks like:
    # [(params: dict, fepouts: list|str), ...]
    worker_params = []
    bulk_calc_cache = {}
    bulk_pointers = {}
    for calc_name in cp.sections():
        params = global_params.copy()
        params['calc_name'] = calc_name
        params['detectEQ'] = True
        params['decorrelate'] = True
        # bound
        params['leg_name'] = 'bound'
        worker_params.append((params.copy(), cp[calc_name]['bound']))

        # bulk
        params['leg_name'] = 'bulk'
        if str(cp[calc_name]['bulk']) not in bulk_calc_cache:
            worker_params.append((params.copy(), cp[calc_name]['bulk']))
            bulk_calc_cache[str(cp[calc_name]['bulk'])] = calc_name
        else:
            prev_calc_name = bulk_calc_cache[str(cp[calc_name]['bulk'])]
            bulk_pointers[calc_name] = prev_calc_name
            # print(f'Skipped duplicate bulk calculation: {calc_name} uses the same bulk as {prev_calc_name}')

    with Pool() as p:
        results = p.map(fep_worker, worker_params)

    # Reorganize results into their individual calc_names
    final = defaultdict(dict)
    for i in range(len(results)):
        params_i, _ = worker_params[i]
        cumulative, per_window, u_nk = results[i]
        calc_name, leg_name = params_i['calc_name'], params_i['leg_name']
        final[calc_name][f'{leg_name} cumulative'] = cumulative
        final[calc_name][f'{leg_name} per_window'] = per_window
        final[calc_name][f'{leg_name} u_nk'] = u_nk

    # TODO: Make all the pretty figures
    for calc_name in cp.sections():
        # Look in our bulk calculation cache as necessary
        if 'bulk cumulative' not in final[calc_name]:
            cumulative_bulk = final[bulk_pointers[calc_name]]['bulk cumulative']
        else:
            cumulative_bulk = final[calc_name]['bulk cumulative']
        RT = global_params['RT']
        dG_bulk = np.round(cumulative_bulk.BAR.f.iloc[-1] * RT, 1)
        error_bulk = np.round(cumulative_bulk.BAR.errors.iloc[-1] * RT, 1)

        # Volumetric restraint contribution
        molar = 1660  # cubic angstroms per particle in a one molar solution
        dG_V = np.round(-float(RT) * np.log(4 / 3 * scipy.pi * float(cp[calc_name]['comradius']) ** 3 / molar), 1)
        error_V = 0.0

        # Process the DBC TI RFEP for each calc_name
        ti_per_window, ti_cumulative = process_dbc_ti(calc_name, cp[calc_name])
        dG_DBC = np.round(ti_cumulative['dG'][1], 1)
        error_DBC = np.round(ti_cumulative['error'][1], 1)

        cumulative = final[calc_name]['bound cumulative']
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
    lsched = np.linspace(1, 0, n_lambdas)
    dbc = safep.make_harmonicWall(FC=0,
                                  targetFC=float(params['rfep_fc']),
                                  targetFE=6,
                                  upperWalls=float(params['dbcwidth']),
                                  targetEQ=500,
                                  numSteps=300000,
                                  name='harmonicwalls2',
                                  schedule=lsched)
    Ls = (data_ti.index.values - 1) // dbc['numSteps']
    Ls[0] = 0
    Ls[Ls == n_lambdas] = n_lambdas - 1  # This is a small hack in case there are extra samples for the last window

    data_ls = np.round([dbc['schedule'][i] for i in Ls], 3)
    data_ti.loc[:, 'L'] = data_ls
    data_ti = data_ti.iloc[1:]

    ti_per_window, ti_cumulative = safep.process_TI(data_ti, dbc, lsched)
    # print(calc_name, 'rfep', ti_cumulative['dG'][1], ti_cumulative['error'][1])
    return ti_per_window, ti_cumulative


def fep_worker(args):
    params, fepouts = args
    fepouts = glob_as_necessary(fepouts)
    u_nk = safep.read_and_process(fepouts, params['temperature'], decorrelate=params['decorrelate'],
                                   detectEQ=params['detectEQ'])
    per_window, cumulative = safep.do_estimation(u_nk)
    dG_site = np.round(cumulative.BAR.f.iloc[-1] * params['RT'], 1)
    error_site = np.round(cumulative.BAR.errors.iloc[-1] * params['RT'], 1)
    # print(f'{params["calc_name"]} {params["leg_name"]}: dG = {dG_site:.2f} kcal/mol, error = {error_site:.2f} kcal/mol')
    return cumulative, per_window, u_nk


def glob_as_necessary(path):
    """Takes a single glob expression or a list of glob expressions, and
    globs them into a list of files. Prefixing with '@' specifies a filename
    containing these things.
    """
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
