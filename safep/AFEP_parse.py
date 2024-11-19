"""
Large datasets can be difficult to parse on a workstation due to inefficiencies 
in the way data is represented for pymbar. When possible, reduce the size of your dataset.
"""
import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from alchemlyb.parsing import namd

import safep

warnings.simplefilter(action="ignore", category=FutureWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--path",
        type=str,
        help="The absolute path to the directory containing the fepout files",
        default=".",
    )
    parser.add_argument(
        "--fepoutre",
        type=str,
        help="A regular expression that matches the fepout files of interest.",
        default="*.fep*",
    )
    parser.add_argument(
        "--replicare",
        type=str,
        help="A regular expression that matches the replica directories",
        default="Replica?",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="The temperature at which the FEP was run.",
        default=303.15,
    )
    parser.add_argument(
        "--detectEQ",
        type=bool,
        help="Flag for automated equilibrium detection.",
        default=True,
    )
    parser.add_argument(
        "--fittingMethod",
        type=str,
        help="Method for fitting the forward-backward discrepancies (hysteresis)."
        + "LS=least squares, ML=maximum likelihood Default: LS",
        default="LS",
    )
    parser.add_argument(
        "--maxSize",
        type=float,
        help="Maximum total file size in GB. This is MUCH less than the required RAM. Default: 1",
        default=1,
    )
    parser.add_argument(
        "--makeFigures",
        type=bool,
        help="Run additional diagnostics and save figures to the directory. default: False",
        default=0,
    )

    args = parser.parse_args()

    dataroot = Path(args.path)
    replica_pattern = args.replicare
    replicas = dataroot.glob(replica_pattern)
    filename_pattern = args.fepoutre

    temperature = args.temperature
    RT = 0.00198720650096 * temperature
    detectEQ = args.detectEQ

    colors = ["blue", "red", "green", "purple", "orange", "violet", "cyan"]
    itcolors = iter(colors)

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

    # # Extract key features from the MBAR fitting and get ΔG
    # Note: alchemlyb operates in units of kT by default.
    # We multiply by RT to convert to units of kcal/mol.

    # # Read and plot number of samples after detecting EQ

    fepruns = {}
    for replica in replicas:
        print(f"Reading {replica}")
        unkpath = replica.joinpath("decorrelated.csv")
        u_nk = None
        if unkpath.is_file():
            print("Found existing dataframe. Reading.")
            u_nk = safep.read_UNK(unkpath)
        else:
            print(
                f"Didn't find existing dataframe at {unkpath}. Checking for raw fepout files."
            )
            fepoutFiles = list(replica.glob(filename_pattern))
            total_size = 0
            for file in fepoutFiles:
                total_size += os.path.getsize(file)
            print(
                f"Will process {len(fepoutFiles)} fepout files."
                + f"\nTotal size:{np.round(total_size/10**9, 2)}GB"
            )

            if len(list(fepoutFiles)) > 0:
                print("Reading fepout files")
                fig, ax = plt.subplots()

                u_nk = namd.extract_u_nk(fepoutFiles, temperature)
                u_nk = u_nk.sort_index(axis=0, level=1).sort_index(axis=1)
                safep.plot_samples(ax, u_nk, color="blue", label="Raw Data")

                if detectEQ:
                    print("Detecting equilibrium")
                    u_nk = safep.detect_equilibrium_u_nk(u_nk)
                    safep.plot_samples(
                        ax, u_nk, color="orange", label="Equilibrium-Detected"
                    )

                plt.savefig(dataroot.joinpath(f"{str(replica)}_FEP_number_of_samples.pdf"))
                safep.save_UNK(u_nk, unkpath)
            else:
                print(f"WARNING: no fepout files found for {replica}. Skipping.")

        if u_nk is not None:
            fepruns[str(replica)] = FepRun(
                u_nk, None, None, None, None, None, None, None, next(itcolors)
            )

    for key, feprun in fepruns.items():
        u_nk = feprun.u_nk
        feprun.perWindow, feprun.cumulative = safep.do_estimation(
            u_nk
        )  # Run the BAR estimator on the fep data
        (
            feprun.forward,
            feprun.forward_error,
            feprun.backward,
            feprun.backward_error,
        ) = safep.do_convergence(
            u_nk
        )  # Used later in the convergence plot'
        feprun.per_lambda_convergence = safep.do_per_lambda_convergence(u_nk)

    toprint = ""
    dGs = []
    errors = []
    for key, feprun in fepruns.items():
        cumulative = feprun.cumulative
        dG = np.round(cumulative.BAR.f.iloc[-1] * RT, 1)
        error = np.round(cumulative.BAR.errors.iloc[-1] * RT, 1)
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
        # # Plot data
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

            textstr = (
                r"$\rm mode=$"
                + f"{np.round(mode,2)}"
                + "\n"
                + fr"$\mu$={np.round(average,2)}"
                + "\n"
                + fr"$\sigma$={np.round(std,2)}"
            )
            props = dict(boxstyle="square", facecolor="white", alpha=1)
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

        fig = None
        for key, feprun in fepruns.items():
            if fig is None:
                fig, axes = safep.plot_general(
                    feprun.cumulative,
                    None,
                    feprun.perWindow,
                    None,
                    RT,
                    hysttype="lines",
                    label=key,
                    color=feprun.color,
                )
                axes[1].legend()
            else:
                fig, axes = safep.plot_general(
                    feprun.cumulative,
                    None,
                    feprun.perWindow,
                    None,
                    RT,
                    fig=fig,
                    axes=axes,
                    hysttype="lines",
                    label=key,
                    color=feprun.color,
                )
            # fig.suptitle(changeAndError)

        # hack to get aggregate data:
        axes[3] = do_agg_data(axes[2], axes[3])

        axes[0].set_title(str(mean) + r"$\pm$" + str(sterr) + " kcal/mol")
        axes[0].legend()
        plt.savefig(dataroot.joinpath("FEP_general_figures.pdf"))

        # # Plot the estimated total change in free energy as a function of simulation time;
        # contiguous subsets starting at t=0 ("Forward") and t=end ("Reverse")

        fig, convAx = plt.subplots(1, 1)

        for key, feprun in fepruns.items():
            convAx = safep.convergence_plot(
                convAx,
                feprun.forward * RT,
                feprun.forward_error * RT,
                feprun.backward * RT,
                feprun.backward_error * RT,
                fwd_color=feprun.color,
                bwd_color=feprun.color,
                errorbars=False,
            )
            convAx.get_legend().remove()

        (forward_line,) = convAx.plot(
            [], [], linestyle="-", color="black", label="Forward Time Sampling"
        )
        (backward_line,) = convAx.plot(
            [], [], linestyle="--", color="black", label="Backward Time Sampling"
        )
        convAx.legend(handles=[forward_line, backward_line])
        ymin = np.min(dGs) - 1
        ymax = np.max(dGs) + 1
        convAx.set_ylim((ymin, ymax))
        plt.savefig(dataroot.joinpath("FEP_convergence.pdf"))

        genfig = None
        for key, feprun in fepruns.items():
            if genfig is None:
                genfig, genaxes = safep.plot_general(
                    feprun.cumulative,
                    None,
                    feprun.perWindow,
                    None,
                    RT,
                    hysttype="lines",
                    label=key,
                    color=feprun.color,
                )
            else:
                genfig, genaxes = safep.plot_general(
                    feprun.cumulative,
                    None,
                    feprun.perWindow,
                    None,
                    RT,
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
        plt.show()
        plt.savefig(dataroot.joinpath("FEP_perLambda_convergence.pdf"))
