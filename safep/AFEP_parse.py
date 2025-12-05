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


@dataclass(slots=True)
class AFEPArguments():
    dataroot: Path
    replica_pattern: str
    filename_pattern: str
    temperature: float
    detectEQ: bool
    makeFigures: bool
    RT_kcal_per_mol: float=None
    replicas: list[str]=None

    @classmethod
    def from_AFEPArgumentParser(cls, parser: AFEPArgumentParser):
        args = parser.parse_args()

        dataroot = Path(args.path)
        replica_pattern = args.replicare
        filename_pattern = args.fepoutre

        detectEQ = args.detectEQ

        return cls(dataroot, replica_pattern, filename_pattern, args.temperature, detectEQ, args.makeFigures)

    def __post_init__(self) -> None:
        self.RT_kcal_per_mol = R / (KILO * calorie) * self.temperature
        self.replicas = [rep.stem for rep in self.dataroot.glob(self.replica_pattern)]


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
    genfig = None
    for key, feprun in fepruns.items():
        if genfig is None:
            genfig, genaxes = initialize_general_figure(
                args.RT_kcal_per_mol, key, feprun)
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


def make_figures(args, fepruns, dGs, mean, sterr):
    fig = None
    for key, feprun in fepruns.items():
        if fig is None:
            fig, axes = initialize_general_figure(
                args.RT_kcal_per_mol, key, feprun)
        else:
            fig, axes = add_to_general_figure(fig, axes, args, key, feprun)

        # hack to get aggregate data:
    axes = add_summary_stats(do_agg_data, mean, sterr, axes)
    fig.savefig(args.dataroot.joinpath("FEP_general_figures.pdf"))

    # # Plot the estimated total change in free energy as a function of simulation time;
    # contiguous subsets starting at t=0 ("Forward") and t=end ("Reverse")

    fig, convAx = do_shared_convergence_plot(args, fepruns, dGs)
    fig.savefig(args.dataroot.joinpath("FEP_convergence.pdf"))

    fig, axes = do_per_lambda_convergence_shared_axes(
        args, fepruns, mean, sterr, axes)
    fig.savefig(args.dataroot.joinpath("FEP_perLambda_convergence.pdf"))


def get_summary_statistics(args, fepruns):
    toprint = ""
    dGs = []
    errors = []
    for key, feprun in fepruns.items():
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
    parser = AFEPArgumentParser()
    args = AFEPArguments.from_AFEPArgumentParser(parser)
    itcolors = iter(COLORS)
    fepruns = process_replicas(args, itcolors)
    summary, dGs, mean, sterr = get_summary_statistics(args, fepruns)
    print(summary)
    if args.makeFigures == 1:
        make_figures(args, fepruns, dGs, mean, sterr)
        plt.show()


if __name__ == "__main__":
    main()
