"""
Large datasets can be difficult to parse on a workstation due to inefficiencies 
in the way data is represented for pymbar. When possible, reduce the size of your dataset.
"""
import argparse
import os
import warnings
from dataclasses import dataclass
from typing import NamedTuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from alchemlyb.parsing import namd

from scipy.constants import R, calorie

import safep

warnings.simplefilter(action="ignore", category=FutureWarning)


@dataclass
class FepRun:
    u_nk: pd.DataFrame
    perWindow: pd.DataFrame
    cumulative: pd.DataFrame
    forward: pd.DataFrame
    forward_error: pd.DataFrame
    backward: pd.DataFrame
    backward_error: pd.DataFrame
    per_lambda_convergence: pd.DataFrame
    color: str

    def __post_init__(self):
        # Run the BAR estimator on the fep data
        self.perWindow, self.cumulative = safep.do_estimation(self.u_nk)
        (
            self.forward,
            self.forward_error,
            self.backward,
            self.backward_error,
        ) = safep.do_convergence(self.u_nk)  # Used later in the convergence plot
        self.per_lambda_convergence = safep.do_per_lambda_convergence(self.u_nk)


def do_agg_data(dataax, plotax):
    """
    Aggregates data from a given matplotlib axis, computes statistical measures,
    and displays them on another axis.

    Parameters:
    dataax (matplotlib.axes.Axes): The axis containing lines with data to be aggregated.
    plotax (matplotlib.axes.Axes): The axis where the statistical summary will be displayed.

    Returns:
    matplotlib.axes.Axes: The plot axis with the statistical summary text added.
    """
    agg_data = []
    lines = dataax.lines
    for line in lines:
        agg_data.append(line.get_ydata())
    flat = np.array(agg_data).flatten()
    kernel = sp.stats.gaussian_kde(flat)
    pdf_x = np.linspace(-1, 1, 1000)
    pdf_y = kernel(pdf_x)
    std = np.std(flat)
    average = np.average(flat)
    temp = pd.Series(pdf_y, index=pdf_x)
    mode = temp.idxmax()

    textstr = (r"$\rm mode=$" + f"{np.round(mode,2)}" + "\n" + fr"$\mu$={np.round(average,2)}" +
               "\n" + fr"$\sigma$={np.round(std,2)}")
    props = {"boxstyle": 'square', "facecolor": 'white', "alpha": 1}
    plotax.text(
        0.175,
        0.95,
        textstr,
        transform=plotax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    return plotax


class AFEPArgumentParser(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            "--path",
            type=str,
            help="The absolute path to the directory containing the fepout files",
            default=".",
        )
        self.add_argument(
            "--fepoutre",
            type=str,
            help="A regular expression that matches the fepout files to be parsed.",
            default="*.fep*",
        )
        self.add_argument(
            "--replicare",
            type=str,
            help="A regular expression that matches the replica directories",
            default="Replica?",
            required=True,
        )
        self.add_argument(
            "--temperature",
            type=float,
            help="The temperature at which the FEP was run.",
            default=303.15,
        )
        self.add_argument(
            "--detectEQ",
            type=bool,
            help="Flag for automated equilibrium detection.",
            default=True,
        )
        self.add_argument(
            "--fittingMethod",
            type=str,
            help="Method for fitting the forward-backward discrepancies (hysteresis)." +
            "LS=least squares, ML=maximum likelihood Default: LS",
            default="LS",
        )
        self.add_argument(
            "--maxSize",
            type=float,
            help="Maximum total file size in GB." +
            "This is MUCH less than the required RAM. Default: 1",
            default=1,
        )
        self.add_argument(
            "--makeFigures",
            type=bool,
            help="Run additional diagnostics and save figures to the directory. default: False",
            default=0,
        )


KILO = 1000
COLORS = ["blue", "red", "green", "purple", "orange", "violet", "cyan"]


def initialize_general_figure(RT_kcal_per_mol, key, feprun):
    fig, axes = safep.plot_general(
        feprun.cumulative,
        None,
        feprun.perWindow,
        None,
        RT_kcal_per_mol,
        hysttype="lines",
        label=key,
        color=feprun.color,
    )
    axes[1].legend()

    return fig, axes

def report_number_and_size_of_fepout_files(fepoutFiles):
    total_size = 0
    for file in fepoutFiles:
        total_size += os.path.getsize(file)
    print(f"Will process {len(fepoutFiles)} fepout files." +
          f"\nTotal size:{np.round(total_size/10**9, 2)}GB")


class AFEPArguments(NamedTuple):
    dataroot: Path
    replica_pattern: str
    replicas: list[Path]
    filename_pattern: str
    temperature: float
    RT_kcal_per_mol: float
    detectEQ: bool
    makeFigures: bool

    @classmethod
    def from_AFEPArgumentParser(cls, parser: AFEPArgumentParser) -> AFEPArgumentParser:
        args = parser.parse_args()

        dataroot = Path(args.path)
        replica_pattern = args.replicare
        replicas = dataroot.glob(replica_pattern)
        filename_pattern = args.fepoutre

        temperature = args.temperature
        args.RT_kcal_per_mol = R / (KILO * calorie) * temperature
        detectEQ = args.detectEQ

        return cls(dataroot, replica_pattern, replicas, filename_pattern, temperature,
                   args.RT_kcal_per_mol, detectEQ, args.makeFigures)


def read_and_decorrelate(args, replica, unkpath, fepoutFiles):
    fig, ax = plt.subplots()

    u_nk = namd.extract_u_nk(fepoutFiles, args.temperature)
    u_nk = u_nk.sort_index(axis=0, level=1).sort_index(axis=1)
    safep.plot_samples(ax, u_nk, color="blue", label="Raw Data")

    if args.detectEQ:
        print("Detecting equilibrium")
        u_nk = safep.detect_equilibrium_u_nk(u_nk)
        safep.plot_samples(ax, u_nk, color="orange", label="Equilibrium-Detected")

    plt.savefig(args.dataroot.joinpath(f"{str(replica)}_FEP_number_of_samples.pdf"))
    safep.save_UNK(u_nk, unkpath)
    return u_nk


def process_replicas(args, itcolors):
    fepruns = {}
    for replica in args.replicas:
        print(f"Reading {replica}")
        unkpath = replica.joinpath("decorrelated.csv")
        u_nk = None
        if unkpath.is_file():
            print("Found existing dataframe. Reading.")
            u_nk = safep.read_UNK(unkpath)
        else:
            print(f"Didn't find existing dataframe at {unkpath}. Checking for raw fepout files.")
            fepoutFiles = list(replica.glob(args.filename_pattern))
            report_number_and_size_of_fepout_files(fepoutFiles)

            if len(list(fepoutFiles)) > 0:
                print("Reading fepout files")
                u_nk = read_and_decorrelate(args, replica, unkpath, fepoutFiles)
            else:
                print(f"WARNING: no fepout files found for {replica}. Skipping.")

        if u_nk is not None:
            fepruns[str(replica)] = FepRun(u_nk, None, None, None, None, None, None, None,
                                           next(itcolors))
    return fepruns


def add_to_general_figure(fig, axes, args, key, feprun):
    fig, axes = safep.plot_general(
                    feprun.cumulative,
                    None,
                    feprun.perWindow,
                    None,
                    args.RT_kcal_per_mol,
                    fig=fig,
                    axes=axes,
                    hysttype="lines",
                    label=key,
                    color=feprun.color,
                )
    
    return fig, axes

def add_summary_stats(do_agg_data, mean, sterr, axes):
    axes[3] = do_agg_data(axes[2], axes[3])

    axes[0].set_title(str(mean) + r"$\pm$" + str(sterr) + " kcal/mol")
    axes[0].legend()
    return axes

def do_shared_convergence_plot(args, fepruns, dGs):
    fig, convAx = plt.subplots(1, 1)

    for key, feprun in fepruns.items():
        convAx = safep.convergence_plot(
                convAx,
                feprun.forward * args.RT_kcal_per_mol,
                feprun.forward_error * args.RT_kcal_per_mol,
                feprun.backward * args.RT_kcal_per_mol,
                feprun.backward_error * args.RT_kcal_per_mol,
                fwd_color=feprun.color,
                bwd_color=feprun.color,
                errorbars=False,
            )
        convAx.get_legend().remove()

    (forward_line,) = convAx.plot([], [],
                                      linestyle="-",
                                      color="black",
                                      label="Forward Time Sampling")
    (backward_line,) = convAx.plot([], [],
                                       linestyle="--",
                                       color="black",
                                       label="Backward Time Sampling")
    convAx.legend(handles=[forward_line, backward_line])
    ymin = np.min(dGs) - 1
    ymax = np.max(dGs) + 1
    convAx.set_ylim((ymin, ymax))

    return fig, convAx

def do_per_lambda_convergence_shared_axes(do_agg_data, initialize_general_figure, args, fepruns, mean, sterr, axes):
    genfig = None
    for key, feprun in fepruns.items():
        if genfig is None:
            genfig, genaxes = initialize_general_figure(args.RT_kcal_per_mol, key, feprun)
        else:
            genfig, genaxes = safep.plot_general(
                    feprun.cumulative,
                    None,
                    feprun.perWindow,
                    None,
                    args.RT_kcal_per_mol,
                    fig=genfig,
                    axes=genaxes,
                    hysttype="lines",
                    label=key,
                    color=feprun.color,
                )
    plt.delaxes(genaxes[0])
    plt.delaxes(genaxes[1])

    genaxes[3] = do_agg_data(axes[2], axes[3])
    genaxes[2].set_title(str(mean) + r"$\pm$" + str(sterr) + " kcal/mol")

    for txt in genfig.texts:
        print(1)
        txt.set_visible(False)
        txt.set_text("")

    return genfig, genaxes

if __name__ == "__main__":
    parser = AFEPArgumentParser()
    args = AFEPArguments.from_AFEPArgumentParser(parser)

    itcolors = iter(COLORS)

    # # Extract key features from the MBAR fitting and get ΔG
    # Note: alchemlyb operates in units of kT by default.
    # We multiply by RT to convert to units of kcal/mol.

    fepruns = process_replicas(args, itcolors)

    toprint = ""
    dGs = []
    errors = []
    for key, feprun in fepruns.items():
        cumulative = feprun.cumulative
        dG = np.round(cumulative.BAR.f.iloc[-1] * args.RT_kcal_per_mol, 1)
        error = np.round(cumulative.BAR.errors.iloc[-1] * args.RT_kcal_per_mol, 1)
        dGs.append(dG)
        errors.append(error)

        changeAndError = f"{key}: \u0394G = {dG}\u00B1{error} kcal/mol\n"
        toprint += changeAndError

    toprint += "\n"
    mean = np.average(dGs)

    # If there are only a few replicas,
    # the MBAR estimated error will be more reliable, albeit underestimated
    if len(dGs) < 3:
        sterr = np.sqrt(np.sum(np.square(errors)))
    else:
        sterr = np.round(np.std(dGs), 1)
    toprint += f"mean: {mean} kcal/mol\n" + f"sterr: {sterr} kcal/mol"
    print(toprint)

    if args.makeFigures == 1:
        fig = None
        for key, feprun in fepruns.items():
            if fig is None:
                fig, axes = initialize_general_figure(args.RT_kcal_per_mol, key, feprun)
            else:
                fig, axes = add_to_general_figure(fig, axes, args, key, feprun)

        # hack to get aggregate data:
        axes = add_summary_stats(do_agg_data, mean, sterr, axes)
        fig.savefig(args.dataroot.joinpath("FEP_general_figures.pdf"))

        # # Plot the estimated total change in free energy as a function of simulation time;
        # contiguous subsets starting at t=0 ("Forward") and t=end ("Reverse")

        fig, convAx = do_shared_convergence_plot(args, fepruns, dGs)
        fig.savefig(args.dataroot.joinpath("FEP_convergence.pdf"))

        fig, axes = do_per_lambda_convergence_shared_axes(do_agg_data, initialize_general_figure, args, fepruns, mean, sterr, axes)
        fig.savefig(args.dataroot.joinpath("FEP_perLambda_convergence.pdf"))

        plt.show()
