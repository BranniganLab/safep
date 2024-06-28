#!/usr/bin/env python3
import argparse
import configparser
import glob
import re
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import pandas as pd
import scipy
import numpy as np
from matplotlib import pyplot as plt

import safep

"""
; Sample configuration file
; DEFAULT must be capitalized for some reason

[DEFAULT]
; These values exist in all other sections
temperature=303.15
figout_path=/tmp
; Since all these calculations share a bulk leg, we define it once here to save on typing
bulk=/nas1/bar/baz/ligdisA*.fepout

[calc1]
bound=/nas1/foo/bar/whatever/namd/disprodA*.fepout
rfep=/nas1/foo/bar/whatever/namd/RFEP.colvars.traj
dbcwidth=5
comradius=6

[calc2]
bound=/nas1/foo/bar/whatever/namd/disprodA*.fepout
rfep=/nas1/foo/bar/whatever/namd/RFEP.colvars.traj
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
    R = scipy.constants.R / (1000 * scipy.constants.calorie)  # gas constant in kcal/(mol K)
    global_params = {'temperature': temperature, 'RT': R * temperature}

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

    # Postprocess (and also process TI)
    for calc_name in config.sections():
        # Look in our bulk calculation cache as necessary. We stored the e
        if 'bulk cumulative' not in final_results[calc_name]:
            cumulative_bulk = final_results[bulk_pointers[calc_name]]['bulk cumulative']
            per_window_bulk = final_results[bulk_pointers[calc_name]]['bulk per_window']
            u_nk_bulk = final_results[bulk_pointers[calc_name]]['bulk u_nk']

        else:
            cumulative_bulk = final_results[calc_name]['bulk cumulative']
            per_window_bulk = final_results[calc_name]['bulk per_window']
            u_nk_bulk = final_results[calc_name]['bulk u_nk']
        RT = global_params['RT']
        dG_bulk = np.round(cumulative_bulk.BAR.f.iloc[-1] * RT, 1)
        error_bulk = np.round(cumulative_bulk.BAR.errors.iloc[-1] * RT, 1)

        # Volumetric restraint contribution
        molar = 1660  # cubic angstroms per particle in a one molar solution
        dG_V = np.round(-float(RT) * np.log(4 / 3 * scipy.pi * float(config[calc_name]['comradius']) ** 3 / molar), 1)
        error_V = 0.0

        # Process the DBC TI RFEP for each calc_name. We don't use worker processes for this because it's typically
        # a relatively small amount of computation, but there's no fundamental reason not to if we choose.
        ti_per_window, ti_cumulative = process_dbc_ti(config[calc_name])
        dG_DBC = np.round(ti_cumulative['dG'][1], 1)
        error_DBC = np.round(ti_cumulative['error'][1], 1)

        dG_site = np.round(final_results[calc_name][f'bound cumulative'].BAR.f.iloc[-1] * RT, 1)
        error_site = np.round(final_results[calc_name][f'bound cumulative'].BAR.errors.iloc[-1] * RT, 1)

        print(f"{calc_name}: Writing figures to {config[calc_name]['outputdir']}")
        plot_general(config[calc_name]['outputdir'],
                     f'{calc_name} bound',
                     final_results[calc_name][f'bound cumulative'],
                     final_results[calc_name][f'bound per_window'],
                     final_results[calc_name][f'bound u_nk'],
                     global_params['RT'])

        plot_general(config[calc_name]['outputdir'],
                     f'{calc_name} bulk',
                     cumulative_bulk,
                     per_window_bulk,
                     u_nk_bulk,
                     global_params['RT'])

        # TODO: Make TI figure size configurable
        # TODO: Close TI figure using plt.close()
        safep.plot_TI(ti_cumulative, ti_per_window, width=6, height=2, fontsize=20)
        plt.savefig(Path(config[calc_name]['outputdir']) / f'{calc_name}_TI_general.pdf')

        dG_binding = dG_bulk + dG_V + dG_DBC - dG_site
        error_binding = np.sqrt(np.sum(np.array([error_bulk, error_V, error_site, error_DBC]) ** 2))
        print(calc_name, f'dG_bulk = {dG_bulk:.2f} (err: {error_bulk:.2f} kcal/mol)')
        print(calc_name, f'dG_V = {dG_V:.2f} (err: {error_V:.2f} kcal/mol)')
        print(calc_name, f'dG_DBC = {dG_DBC:.2f} (err: {error_DBC:.2f} kcal/mol)')
        print(calc_name, f'dG_site = {dG_site:.2f} (err: {error_site:.2f} kcal/mol)')
        print(calc_name, f'dG_binding = {dG_binding:.2f} (err: {error_binding:.2f} kcal/mol)')

        plot_titration_curve(config[calc_name]['outputdir'],
                             calc_name,
                             global_params['RT'],
                             dG_binding,
                             error_binding)

def process_dbc_ti(params):
    """Process the DBC TI output and return per-window and cumulative energy curves."""
    with open(params['rfep']) as f:
        first_line = f.readline()
    columns = re.split(' +', first_line)[1:-1]
    data_ti = pd.read_csv(params['rfep'], delim_whitespace=True, names=columns, comment='#', index_col=0)
    data_ti = data_ti[data_ti.index >= 1000][1:]
    data_ti.index = data_ti.index - 1000

    n_lambdas = int(params['rfep_n'])

    safep.make_harmonicWall(FC=float(params['rfep_fc']),
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


def plot_general(out_path, out_prefix: str, cumulative, per_window, u_nk, RT: float, figsize=(6, 3), pdf_type='KDE',
                 fontsize=20):
    out_path = Path(out_path)

    def guess_ylim(data):
        return np.min(data), np.max(data)

    # Cumulative change in kcal/mol
    cumul_fig, cumul_ax = plt.subplots(figsize=figsize)
    cumul_ax.errorbar(cumulative.index, cumulative.BAR.f * RT, yerr=cumulative.BAR.errors, marker=None, linewidth=1)
    cumul_ax.set(ylabel=r'Cumulative $\mathrm{\Delta} G_{\lambda}$' + '\n(kcal/mol)',
                 ylim=guess_ylim(cumulative.BAR.f * RT))
    cumul_fig.tight_layout()
    cumul_fig.savefig(out_path / f'{out_prefix}_cumulative.pdf')
    plt.close(cumul_fig)

    # Per-window change in kcal/mol
    each_fig, each_ax = plt.subplots(figsize=figsize)
    each_ax.errorbar(per_window.index, per_window.BAR.df * RT, yerr=per_window.BAR.ddf, marker=None, linewidth=1)
    each_ax.plot(per_window.index, per_window.EXP.dG_f * RT, marker=None, linewidth=1, alpha=0.5)
    each_ax.errorbar(per_window.index, -per_window.EXP.dG_b * RT, marker=None, linewidth=1, alpha=0.5)
    each_ax.set(ylabel=r'$\mathrm{\Delta} G_\lambda$' + '\n' + r'$\left(kcal/mol\right)$',
                ylim=guess_ylim(per_window.EXP.dG_f * RT))
    each_ax.set_ylim((-0.4, 0.4))
    each_fig.tight_layout()
    each_fig.savefig(out_path / f'{out_prefix}_per_window_change.pdf')
    plt.close(each_fig)

    # Hysteresis Plots
    diff = per_window.EXP['difference']
    hyst_fig, hyst_axs = plt.subplots(nrows=1, ncols=2, figsize=figsize, width_ratios=(3, 1), sharey=True)
    hyst_ax, pdf_ax = hyst_axs
    hyst_ax.vlines(per_window.index, np.zeros(len(per_window)), diff, label="fwd - bwd", linewidth=2)
    hyst_ax.set(ylabel=r'$\delta_\lambda$ (kcal/mol)', ylim=(-1, 1))
    hyst_ax.set_xlabel(xlabel=r'$\lambda$', fontsize=fontsize)

    if pdf_type == 'KDE':
        kernel = scipy.stats.gaussian_kde(diff)
        pdf_x = np.linspace(-1, 1, 1000)
        pdf_y = kernel(pdf_x)
        pdf_ax.plot(pdf_y, pdf_x, label='KDE')
    elif pdf_type == 'Histogram':
        pdf_y, pdf_x = np.histogram(diff, density=True)
        pdf_x = pdf_x[:-1] + (pdf_x[1] - pdf_x[0]) / 2
        pdf_ax.plot(pdf_y, pdf_x, label="Estimated Distribution")
    else:
        raise f"Error: PDF type {pdf_type} not recognized"

    pdf_ax.set_xlabel(pdf_type, fontsize=fontsize)

    std = np.std(diff)
    mean = np.average(diff)
    temp = pd.Series(pdf_y, index=pdf_x)
    mode = temp.idxmax()

    textstr = (r"$\rm mode=$" + f"{np.round(mode, 2)}" + "\n" + fr"$\mu$={np.round(mean, 2)}" + "\n" +
               fr"$\sigma$={np.round(std, 2)}")
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    pdf_ax.text(0.15, 0.95, textstr, transform=pdf_ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    hyst_fig.tight_layout()
    hyst_fig.savefig(out_path / f'{out_prefix}_hysteresis.pdf')
    plt.close(hyst_fig)

    per_lambda_convergence = safep.do_per_lambda_convergence(u_nk)
    plc_fig, plc_ax = plt.subplots(figsize=figsize)
    plc_ax.errorbar(per_lambda_convergence.index, per_lambda_convergence.BAR.df * RT)
    plc_ax.set_xlabel(r"$\lambda$")
    plc_ax.set_ylabel(r"$D_{last-first}$ (kcal/mol)")
    plc_ax.set_ylim((-0.1, 0.1))
    plc_fig.tight_layout()
    plc_fig.savefig(out_path / f'{out_prefix}_per_lambda_convergence.pdf')
    plt.close(plc_fig)

    # Convergence as a function of simulation time
    tc_fig, tc_ax = plt.subplots(figsize=figsize)
    forward, forward_error, backward, backward_error = safep.do_convergence(u_nk)  # Used later in the convergence plot
    safep.convergence_plot(tc_ax, forward * RT, forward_error * RT, backward * RT, backward_error * RT)
    # tc_ax.set_xlim(0, 1.1)
    tc_fig.tight_layout()
    tc_fig.savefig(out_path / f'{out_prefix}_per_window_convergence.pdf')
    plt.close(tc_fig)


def plot_titration_curve(out_path, out_prefix: str, RT: float, dG_binding: float, error_binding: float):
    def P_bind(K, L):
        return L / (K + L)

    def Kd(dG):
        return np.exp(dG / RT) * 1e6

    concentrations = np.logspace(-1, 4, 100)
    K = Kd(dG_binding)
    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)
    fig, ax = plt.subplots(figsize=(10, 6.1))
    ax.plot(concentrations, P_bind(K, concentrations), label='Binding Curve')
    ax.fill_between(concentrations, P_bind(Kd(dG_binding - error_binding * 1.96), concentrations),
                    P_bind(Kd(dG_binding + error_binding * 1.96), concentrations), alpha=0.25,
                    label='95% Confidence Interval')
    plt.xscale('log')
    ax.set_xlabel('Concentration of propofol ' + r'($\mathrm{µ}$M)', fontsize=20)
    ax.set_ylabel('Fraction of sites occupied', fontsize=20)
    ax.vlines(K, 0, 1, linestyles='dashed', color='black', label='Dissociation Constant')
    ax.legend(loc='lower right', fontsize=20 * 0.75)

    plt.savefig(Path(out_path) / f'{out_prefix}_titration_curve.pdf')
    plt.close(fig)

    K_lo, K_hi = Kd(dG_binding - error_binding * 1.96), Kd(dG_binding + error_binding * 1.96)
    print(f'{out_prefix} Kd = {K:.1f} uM (95% CI: {K_lo:.1f}–{K_hi:.1f})')


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
    if type(path) is str:
        if path.startswith('@'):
            with open(path[1:], 'r') as f:
                return glob_as_necessary(f.readlines())
        else:
            return glob.glob(path)
    elif type(path) is list:
        files = []
        for filename in path:
            files.extend(glob.glob(filename.strip()))
        return files
    else:
        return []


if __name__ == '__main__':
    main()
