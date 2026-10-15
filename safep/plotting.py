# Import block
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib.axes._axes import Axes
from numbers import Number
from numpy import ndarray
from pandas.core.frame import DataFrame

from .processing import get_n_samples


class SampleAxes(matplotlib.axes.Axes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_ylabel("Number of Samples")
        self.set_xlabel(r"$\lambda$")
        self.set_yscale("log")

    def plot_samples(self, u_nk: DataFrame, color: str="blue", label: str="Raw Data"):
        """
        Plot the number of samples against the lambda values.

        Args:
            u_nk (pd.DataFrame): The input data frame containing the samples and lambda values.
            color (str, optional): The color of the plot line. Defaults to 'blue'.
            label (str, optional): The label for the plot legend. Defaults to 'Raw Data'.

        Returns:
            matplotlib.axes.Axes: The axes object after plotting the data.
        """
        samples = get_n_samples(u_nk)
        self.plot(samples, color=color, label=label)
        self.legend()


def convergence_plot(
    theax: Axes,
    fs: ndarray,
    ferr: ndarray,
    bs: ndarray,
    berr: ndarray,
    fwd_color:str ="#0072B2",
    bwd_color: str="#D55E00",
    fwd_legend: str="",
    bwd_legend: str="",
    fontsize: int=12,
    errorbars: bool=True,
):
    """
    Convergence plot. Does the convergence plotting.
    Returns: a pyplot
    """
    if fwd_legend=="":
        fwd_legend = fwd_color
        bwd_legend = bwd_color

    lower = fs[-1] - ferr[-1]
    upper = fs[-1] + ferr[-1]
    theax.fill_between([0, 1], [lower, lower], [upper, upper], color="gray", alpha=0.1)

    fwdparams = {
        "marker": "o",
        "linewidth": 1,
        "color": fwd_color,
        "markerfacecolor": "white",
        "markeredgewidth": 1,
        "markeredgecolor": fwd_color,
        "ms": 5,
    }
    bwdparams = {
        "marker": "o",
        "linewidth": 1,
        "color": bwd_color,
        "markerfacecolor": "white",
        "markeredgewidth": 1,
        "markeredgecolor": bwd_color,
        "ms": 5,
        "linestyle": "--",
    }

    xfwd = np.arange(len(fs)) / len(fs) + 0.1
    xbwd = np.arange(len(bs)) / len(fs) + 0.1

    if errorbars:
        theax.errorbar(xfwd, fs, yerr=ferr, **fwdparams)
        theax.errorbar(xbwd, bs, yerr=berr, **bwdparams)
    else:
        theax.errorbar(xfwd, fs, **fwdparams)
        theax.errorbar(xbwd, bs, **bwdparams)

    theax.xaxis.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    final_mean = fs[-1]
    theax.axhline(y=final_mean, linestyle="-.", color="gray")
    theax.set_ylim((final_mean - 0.75, final_mean + 0.75))

    theax.plot(
        0, final_mean, linewidth=1, color=fwd_legend, label="Forward Time Sampling"
    )
    theax.plot(
        0,
        final_mean,
        linewidth=1,
        color=bwd_legend,
        linestyle="--",
        label="Backward Time Sampling",
    )
    theax.set_xlabel("Fraction of Simulation Time", fontsize=fontsize)
    theax.set_ylabel(r"Total $\mathrm{\Delta} G$ (kcal/mol)", fontsize=fontsize)
    theax.legend()
    return theax


def plot_general(
    cumulative: pd.DataFrame,
    cumulative_ylim: List[Number],
    per_window: pd.DataFrame,
    per_window_ylim: List[Number],
    RT: Number,
    width: Number = 8,
    height: Number = 4,
    pdf_type: str = "KDE",
    fontsize: int = 12,
    fig: matplotlib.pyplot.Figure | None = None,
    axes: List[Axes] | None = None,
    hysttype: str = "classic",
    label: str | None = None,
    color: str = "blue",
    errorbars: bool = True,
    xlim: List[Number] = [-1, 1],
):
    """
    Plot the general analysis of cumulative and per-window changes in delta-G.

    Arguments:
    - cumulative: A pandas DataFrame containing the cumulative changes in delta-G.
    - cumulativeYlim: The y-axis limits for the cumulative plot.
    - perWindow: A pandas DataFrame containing the per-window changes in delta-G.
    - perWindowYlim: The y-axis limits for the per-window plot.
    - RT: The gas constant times the temperature.
    - width: The width of the figure.
    - height: The height of the figure.
    - PDFtype: The type of probability density function to plot (either 'KDE' or 'Histogram').
    - fontsize: The font size of the plot labels.
    - fig: The figure object to use for plotting.
    - axes: The axes objects to use for plotting.
    - hysttype: The type of hysteresis plot to use (either 'classic' or 'lines').
    - label: The label for the plot.
    - color: The color of the plot.
    - errorbars: Whether to include error bars in the per-window plot.

    Returns:
    - fig: The figure object.
    - axes: The axes objects.
    """
    if fig is None and axes is None:
        fig, axes = plt.subplots(
            3, 2, sharex="col", sharey="row", gridspec_kw={"width_ratios": [2, 1]}
        )
        ((cumul_ax, del1), (each_ax, del2), (hyst_ax, pdf_ax)) = axes

        fig.delaxes(del1)
        fig.delaxes(del2)
    else:
        cumul_ax, each_ax, hyst_ax, pdf_ax = axes

    # Cumulative change in kcal/mol
    cumul_ax.errorbar(
        cumulative.index,
        cumulative.BAR.f * RT,
        yerr=cumulative.BAR.errors,
        marker=None,
        linewidth=1,
        label=label,
        color=color,
    )
    cumul_ax.set(
        ylabel=r"Cumulative $\mathrm{\Delta} G_{\lambda}$" + "\n(kcal/mol)",
        ylim=cumulative_ylim,
    )

    # Per-window change in kcal/mol
    if errorbars:
        each_ax.errorbar(
            per_window.index,
            per_window.BAR.df * RT,
            yerr=per_window.BAR.ddf,
            marker=None,
            linewidth=1,
            alpha=0.5,
            color=color,
            label="BAR",
        )
    else:
        each_ax.plot(
            per_window.index,
            per_window.BAR.df * RT,
            marker=None,
            linewidth=1,
            color=color,
            label="BAR",
        )

    each_ax.plot(
        per_window.index,
        per_window.EXP.dG_f * RT,
        marker=None,
        linewidth=1,
        alpha=0.5,
        color=color,
        label="forward EXP",
    )
    each_ax.plot(
        per_window.index,
        -per_window.EXP.dG_b * RT,
        marker=None,
        linewidth=1,
        alpha=0.5,
        linestyle="--",
        color=color,
        label="backward EXP",
    )

    if each_ax.get_legend() is None:
        (line_bar,) = each_ax.plot(
            [], [], marker=None, linewidth=1, color="k", label="BAR"
        )
        (line_fwd,) = each_ax.plot(
            [], [], marker=None, linewidth=1, alpha=0.5, color="k", label="forward EXP"
        )
        (line_bwd,) = each_ax.plot(
            [],
            [],
            marker=None,
            linewidth=1,
            alpha=0.5,
            linestyle="--",
            color="k",
            label="backward EXP",
        )
        each_ax.legend(handles=[line_bar, line_fwd, line_bwd])

    each_ax.set(
        ylabel=r"$\mathrm{\Delta} G_\lambda$"
        + "\n"
        + r"$\left(\mathrm{kcal}/\mathrm{mol}\right)$",
        ylim=per_window_ylim,
    )

    # Hysteresis Plots
    hyst_ax, pdf_ax = plot_hysteresis(
        (hyst_ax, pdf_ax), per_window, hysttype, color, fontsize, pdf_type, xlim
    )

    fig.set_figwidth(width)
    fig.set_figheight(height * 3)
    fig.tight_layout()

    for ax in [cumul_ax, each_ax, hyst_ax, pdf_ax]:
        ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)

    return fig, [cumul_ax, each_ax, hyst_ax, pdf_ax]


def add_hyst_textbox(diff: ndarray, pdf_x: ndarray, pdf_y: ndarray, pdf_ax: Axes, xoffset: Number=0.15, yoffset: Number=0.95, fs: int=14):
    """
    Add a textbox to a probability density function (PDF) plot.

    Parameters:
    - diff (array-like): The differences between forward and backward estimates.
    - pdf_x (array-like): The x-values of the PDF plot.
    - pdf_y (array-like): The y-values of the PDF plot.
    - pdf_ax (matplotlib.axes.Axes): The axes object of the PDF plot.
    - xoffset, yoffset (floats): the offsets for placing the textbox (default:0.15,0.95, respectively)
    - fs: the fontsize (default:14)

    Returns:
    - pdf_ax (matplotlib.axes.Axes): The updated axes object with the textbox added.

    Description:
    This function adds a textbox to a probability density function (PDF) plot. The textbox displays the mode, mean, and standard deviation of the differences between forward and backward estimates. The mode is calculated as the x-value at which the PDF reaches its maximum. The mean and standard deviation are calculated using the numpy functions np.average() and np.std(), respectively.

    The textbox is positioned at the coordinates (xoffset, yoffset) relative to the axes object's transformation. The text is formatted using LaTeX syntax and includes the mode, mean, and standard deviation values.

    Example:
    diff = [0.1, 0.2, 0.3, 0.4, 0.5]
    pdf_x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    pdf_y = [0.2, 0.4, 0.6, 0.8, 1.0, 0.8]
    pdf_ax = plt.gca()

    pdf_ax = add_hyst_textbox(diff, pdf_x, pdf_y, pdf_ax)
    plt.show()
    """
    std = np.std(diff)
    mean = np.average(diff)
    temp = pd.Series(pdf_y, index=pdf_x)
    mode = temp.idxmax()

    textstr = (
        rf"$\rm mode=${np.round(mode,2)}"
        + "\n"
        + rf"$\mu$={np.round(mean,2)}"
        + "\n"
        + rf"$\sigma$={np.round(std,2)}"
    )
    props = dict(boxstyle="square", facecolor="white", alpha=0.5)
    pdf_ax.text(
        xoffset,
        yoffset,
        textstr,
        transform=pdf_ax.transAxes,
        fontsize=fs,
        verticalalignment="top",
        bbox=props,
    )

    return pdf_ax


def plot_hysteresis(
    axes: tuple[Axes,Axes],
    per_window: DataFrame,
    hysttype: str,
    color: str="blue",
    fontsize: int=12,
    pdf_type: str="KDE",
    xlim: List[float]=[-1, 1],
    textbox: bool=True,
):
    hyst_ax, pdf_ax = axes

    diff = per_window.EXP["difference"]
    assert hysttype in [
        "classic",
        "lines",
    ], f"ERROR: I don't know how to plot hysttype={hysttype}"
    if hysttype == "classic":
        hyst_ax.vlines(
            per_window.index,
            np.zeros(len(per_window)),
            diff,
            label="fwd - bwd",
            linewidth=2,
            color=color,
        )
    elif hysttype == "lines":
        hyst_ax.plot(
            per_window.index, diff, label="fwd - bwd", linewidth=1, color=color
        )

    ytxt = r"$\delta_\lambda$ (kcal/mol)"
    hyst_ax.set_ylabel(ylabel=ytxt, fontsize=fontsize)
    hyst_ax.set_ylim(xlim)

    xtxt = r"$\lambda$"
    hyst_ax.set_xlabel(xlabel=xtxt, fontsize=fontsize)

    if pdf_type == "KDE":
        kernel = sp.stats.gaussian_kde(diff)
        pdf_x = np.linspace(xlim[0], xlim[1], 1000)
        pdf_y = kernel(pdf_x)
        pdf_ax.plot(pdf_y, pdf_x, label="KDE", color=color)
    elif pdf_type == "Histogram":
        pdf_y, pdf_x = np.histogram(diff, density=True)
        pdf_x = pdf_x[:-1] + (pdf_x[1] - pdf_x[0]) / 2
        pdf_ax.plot(pdf_y, pdf_x, label="Estimated Distribution", color=color)
    else:
        raise ValueError(f"PDFtype {pdf_type} not recognized")

    pdf_ax.set_xlabel(pdf_type, fontsize=fontsize)

    if textbox:
        pdf_ax = add_hyst_textbox(diff, pdf_x, pdf_y, pdf_ax)

    return hyst_ax, pdf_ax
