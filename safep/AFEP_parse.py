"""
Large datasets can be difficult to parse on a workstation due to inefficiencies 
in the way data is represented for pymbar. When possible, reduce the size of your dataset.
"""
import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from scipy.constants import R, calorie

import safep
from safep.fepruns import process_replicas

warnings.simplefilter(action="ignore", category=FutureWarning)


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
    """Dedicated CLI argument parser for AFEP.

    Attributes:
        path (str|Path): root path for data folder.
        fepoutre (str): regex for fepout files
        replicare (str): regex for replica directories
        temperature (float): the temperature at which the simulation was run (K)
        detect_equilibrium (bool): Flag to run automated equilibrium detection and downsampling
        fittingMethod (str): Method for fitting forward-backward discrepancies. Untested.
        max_size (float): UNUSED. Maximum size of data to parse. Default 1GB
            Note: this should be much less than the total RAM available.
        make_figures (bool): Whether or not to generate figures. Default False.
    """

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
            "--detect_equilibrium",
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
            "--max_size",
            type=float,
            help="Maximum total file size in GB." +
            "This is MUCH less than the required RAM. Default: 1",
            default=1,
        )
        self.add_argument(
            "--make_figures",
            type=bool,
            help="Run additional diagnostics and save figures to the directory. default: False",
            default=0,
        )


KILO = 1000
COLORS = ["blue", "red", "green", "purple", "orange", "violet", "cyan"]


def initialize_general_figure(RT_kcal_per_mol, key, feprun):
    """Create a new general figure

    Args:
        RT_kcal_per_mol (float): RT in kcal/mol
        key (str): The key of the initial dataset
        feprun (safep.Feprun): the feprun to plot

    Returns:
        Tuple(Fig,Axes): The figure and axes
    """
    fig, axes = safep.plot_general(
        feprun.cumulative,
        None,
        feprun.per_window,
        None,
        RT_kcal_per_mol,
        hysttype="lines",
        label=key,
        color=feprun.color,
    )
    axes[1].legend()

    return fig, axes


@dataclass(slots=True)
class AFEPArguments():
    """A more readable/testable format for CLI arguments"""
    dataroot: Path
    replica_pattern: str
    filename_pattern: str
    temperature: float
    detect_equilibrium: bool
    make_figures: bool
    RT_kcal_per_mol: float=None
    replicas: list[str]=None

    @classmethod
    def from_AFEPArgumentParser(cls, parser: AFEPArgumentParser):
        """Unpack arguments from an AFEPArgumentParser object.

        Args:
            parser (AFEPArgumentParser): An AFEPArgumentParser object.

        Returns:
            AFEPArguments: The unpacked arguments.
        """
        args = parser.parse_args()

        dataroot = Path(args.path)
        replica_pattern = args.replicare
        filename_pattern = args.fepoutre

        detect_equilibrium = args.detect_equilibrium

        return cls(dataroot,
                   replica_pattern,
                   filename_pattern,
                   args.temperature,
                   detect_equilibrium,
                   args.make_figures)

    def __post_init__(self) -> None:
        """Get RT and standardize replica names"""
        self.RT_kcal_per_mol = R / (KILO * calorie) * self.temperature
        self.replicas = [rep.stem for rep in self.dataroot.glob(self.replica_pattern)]


def add_to_general_figure(fig, axes, args, key, feprun):
    """Add another replica to an existing figure.

    Args:
        fig (matplotlib.figure.Figure): The figure to add to.
        axes (matplotlib.axes._subplots.Axes): The axes to add to.
        args (argparse.Namespace): The parsed command line arguments.
        key (str): The key to use for the figure.
        feprun (safep.Feprun): The feprun replica to use.

    Returns:
        tuple(Fig, Axes): The figure and axes that were modified
    """
    fig, axes = safep.plot_general(
        feprun.cumulative,
        None,
        feprun.per_window,
        None,
        args.RT_kcal_per_mol,
        fig=fig,
        axes=axes,
        hysttype="lines",
        label=key,
        color=feprun.color,
    )

    return fig, axes


def add_summary_stats(mean, sterr, axes):
    """Add summary statistics to a general safep figure.

    Args:
        mean (float): the mean free energy across replicas.
        sterr (float): the standard deviation or standard error
            of the free energy across replicas.
        axes (matplotlib.axes): matplotlib axes object

    Returns:
        Axes: the modified matplotlib axes object
    """
    axes[3] = do_agg_data(axes[2], axes[3])

    axes[0].set_title(str(mean) + r"$\pm$" + str(sterr) + " kcal/mol")
    axes[0].legend()
    return axes


def do_shared_convergence_plot(args, fepruns, dGs):
    """Make the convergence plot (reverse and forward cumulative averages)

    Args:
        args (AFEPArguments): command line arguments
        fepruns (dict): fepruns dictionary
        dGs (list): list of free energies

    Returns:
        tuple(Fig, Axes): figure and axes with FCA/RCA convergence

    """
    fig, conv_ax = plt.subplots(1, 1)

    for _, feprun in fepruns.items():
        conv_ax = safep.convergence_plot(
            conv_ax,
            feprun.forward * args.RT_kcal_per_mol,
            feprun.forward_error * args.RT_kcal_per_mol,
            feprun.backward * args.RT_kcal_per_mol,
            feprun.backward_error * args.RT_kcal_per_mol,
            fwd_color=feprun.color,
            bwd_color=feprun.color,
            errorbars=False,
        )
        conv_ax.get_legend().remove()

    (forward_line,) = conv_ax.plot([], [],
                                  linestyle="-",
                                  color="black",
                                  label="Forward Time Sampling")
    (backward_line,) = conv_ax.plot([], [],
                                   linestyle="--",
                                   color="black",
                                   label="Backward Time Sampling")
    conv_ax.legend(handles=[forward_line, backward_line])
    ymin = np.min(dGs) - 1
    ymax = np.max(dGs) + 1
    conv_ax.set_ylim((ymin, ymax))

    return fig, conv_ax


def do_per_lambda_convergence_shared_axes(args, fepruns, mean, sterr, axes):
    """Plot lambda convergence for all replicas.

    Args:
        args (AFEPArguments): command line arguments
        fepruns (dict): fepruns dictionary
        mean (float): mean value across replicas
        sterr (float): sterr value across replicas
        axes (list): matplotlib axes list

    Returns:
        tuple[Fig, Axes]: figure and axes objects with the per lambda convergence
    """
    genfig = None
    for key, feprun in fepruns.items():
        if genfig is None:
            genfig, genaxes = initialize_general_figure(
                args.RT_kcal_per_mol, key, feprun)
        else:
            genfig, genaxes = safep.plot_general(
                feprun.cumulative,
                None,
                feprun.per_window,
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


def make_figures(args, fepruns, dGs, mean, sterr) -> None:
    """Make general SAFEP figures and convergence plots

    Args:
        args (AFEPArguments): arguments from argparse
        fepruns (dict): fepruns dictionary
        dGs (list): list of free energies
        mean (float): mean free energy across replicas
        sterr (float): sterr free energy across replicas

    Returns:
        None

    Side effects:
        Saves FEP_general_figures.pdf, FEP_convergence.pdf, and FEP_perLambda_convergence.pdf
    """
    fig, axes = do_general_figures_plot(args, fepruns, mean, sterr)
    fig.savefig(args.dataroot.joinpath("FEP_general_figures.pdf"))

    # # Plot the estimated total change in free energy as a function of simulation time;
    # contiguous subsets starting at t=0 ("Forward") and t=end ("Reverse")

    fig, _ = do_shared_convergence_plot(args, fepruns, dGs)
    fig.savefig(args.dataroot.joinpath("FEP_convergence.pdf"))

    fig, axes = do_per_lambda_convergence_shared_axes(
        args, fepruns, mean, sterr, axes)
    fig.savefig(args.dataroot.joinpath("FEP_perLambda_convergence.pdf"))


def do_general_figures_plot(args, fepruns, mean, sterr):
    """plot general SAFEP figures

    Args:
        args (AFEPArguments): arguments from argparse
        fepruns (dict): fepruns dictionary
        mean (float): mean free energy across replicas
        sterr (float): sterr free energy across replicas

    Returns:
        fig, axes: figure and axes objects containing the figure panels
        """
    fig = None
    for key, feprun in fepruns.items():
        if fig is None:
            fig, axes = initialize_general_figure(
                args.RT_kcal_per_mol, key, feprun)
        else:
            fig, axes = add_to_general_figure(fig, axes, args, key, feprun)

        # hack to get aggregate data:
    axes = add_summary_stats(mean, sterr, axes)
    fig.tight_layout()
    return fig, axes


def get_summary_statistics(args, fepruns):
    """Extract summary statistics from fepruns.

    Args:
        args (AFEPArguments): parsed command line arguments
        fepruns (dict): dict of fepruns

    Returns:
        tuple[str, list[float], float, float]: a pretty print string, individual delta Gs,
        and the mean and standard error across replicas
    """
    toprint = ""
    dGs = []
    errors = []
    for key, feprun in sorted(fepruns.items()):
        cumulative = feprun.cumulative
        dG = np.round(cumulative.BAR.f.iloc[-1] * args.RT_kcal_per_mol, 1)
        error = np.round(
            cumulative.BAR.errors.iloc[-1] * args.RT_kcal_per_mol, 1)
        dGs.append(dG)
        errors.append(error)

        change_and_error = f"{key}: \u0394G = {dG}\u00B1{error} kcal/mol\n"
        toprint += change_and_error

    toprint += "\n"
    mean = np.average(dGs)

    # If there are only a few replicas,
    # the MBAR estimated error will be more reliable, albeit underestimated
    if len(dGs) < 3:
        sterr = np.sqrt(np.sum(np.square(errors)))
    else:
        sterr = np.round(np.std(dGs), 1)
    toprint += f"mean: {mean} kcal/mol\n" + f"sterr: {sterr} kcal/mol"
    if np.isnan(mean):
        raise RuntimeError("Free energy average is NaN")
    return toprint, dGs, mean, sterr


def main():
    """Main function for parsing fep data and calculating free energy of (de)coupling
    """
    parser = AFEPArgumentParser()
    args = AFEPArguments.from_AFEPArgumentParser(parser)
    itcolors = iter(COLORS)
    fepruns = process_replicas(args, itcolors)
    summary, dGs, mean, sterr = get_summary_statistics(args, fepruns)
    print(summary)
    if args.make_figures == 1:
        make_figures(args, fepruns, dGs, mean, sterr)
        plt.show()


if __name__ == "__main__":
    main()
