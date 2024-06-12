# Import block
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd

from .helpers import *
from .processing import get_n_samples


def plot_samples(theax, u_nk, color='blue', label='Raw Data'):
    """
    Plot the number of samples against the lambda values.

    Args:
        theax (matplotlib.axes.Axes): The axes object to plot on.
        u_nk (pd.DataFrame): The input data frame containing the samples and lambda values.
        color (str, optional): The color of the plot line. Defaults to 'blue'.
        label (str, optional): The label for the plot legend. Defaults to 'Raw Data'.

    Returns:
        matplotlib.axes.Axes: The axes object after plotting the data.
    """
    samples = get_n_samples(u_nk)
    theax.plot(samples, color=color, label=label)
    plt.yscale('log')
    theax.set_ylabel('Number of Samples')
    theax.set_xlabel(r'$\lambda$')
    theax.legend()
    return theax

def do_conv_plot(ax, X, fs, ferr, fwd_color, label=None):
    '''
    A minimalist convergence plot
    Arguments: ax[is], X (the X values), fs (forward estimates), ferr (forward errors), fwdColor (the forward color), label text
    Returns: ax[is]
    '''
    ax.errorbar(X, fs, yerr=ferr, marker=None, linewidth=1, color=fwd_color, markerfacecolor='white', markeredgewidth=1, markeredgecolor=fwd_color, ms=5, label=label)
    return ax


def convergence_plot(theax,
                     fs,
                     ferr,
                     bs,
                     berr,
                     fwd_color='#0072B2',
                     bwd_color='#D55E00',
                     fwd_legend=None,
                     bwd_legend=None,
                     fontsize=12,
                     errorbars=True):
    '''
    Convergence plot. Does the convergence plotting.
    Returns: a pyplot
    '''
    if not fwd_legend:
        fwd_legend=fwd_color
        bwd_legend=bwd_color

    lower = fs[-1]-ferr[-1]
    upper = fs[-1]+ferr[-1]
    theax.fill_between([0,1],[lower, lower], [upper, upper], color='gray', alpha=0.1)

    fwdparams = {'marker':'o',
                'linewidth':1,
                'color':fwd_color,
                'markerfacecolor':'white',
                'markeredgewidth':1,
                'markeredgecolor':fwd_color,
                'ms':5,
                }
    bwdparams = {'marker':'o',
                'linewidth':1,
                'color':bwd_color,
                'markerfacecolor':'white',
                'markeredgewidth':1,
                'markeredgecolor':bwd_color,
                'ms':5,
                'linestyle':'--'}

    xfwd = np.arange(len(fs))/len(fs)+0.1
    xbwd = np.arange(len(bs))/len(fs)+0.1

    if errorbars:
        theax.errorbar(xfwd, fs, yerr=ferr, **fwdparams)
        theax.errorbar(xbwd, bs, yerr=berr, **bwdparams)
    else:
        theax.errorbar(xfwd, fs, **fwdparams)
        theax.errorbar(xbwd, bs, **bwdparams)


    theax.xaxis.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    final_mean = fs[-1]
    theax.axhline(y= final_mean, linestyle='-.', color='gray')
    theax.set_ylim((final_mean-0.75, final_mean+0.75))

    theax.plot(0, final_mean, linewidth=1, color=fwd_legend, label='Forward Time Sampling')
    theax.plot(0, final_mean, linewidth=1, color=bwd_legend, linestyle='--', label='Backward Time Sampling')
    theax.set_xlabel('Fraction of Simulation Time', fontsize=fontsize)
    theax.set_ylabel(r'Total $\mathrm{\Delta} G$ (kcal/mol)', fontsize=fontsize)
    theax.legend()
    return theax
    
    
def fb_discrepancy_plot(l_mid, dG_f, dG_b):
    '''
    Plot the forward-backward discrepancy ("hysteresis")
    Arguments: l_mid (lambda window midpoints), dG_f (forward estimates), dG_b (backward estimates)
    Returns: a pyplot
    '''
    plt.vlines(l_mid, np.zeros(len(l_mid)), dG_f + np.array(dG_b), label="fwd - bwd", linewidth=3)
    plt.legend()
    plt.title('Fwd-bwd discrepancies by lambda')
    plt.xlabel('Lambda')
    plt.ylabel('Diff. in delta-G')
    #Return the figure
    return plt.gca() 

def fb_discrepancy_hist(dG_f, dG_b):
    '''
    Plot the distribution of the hystersis
    Arguments: dG_f, dG_b
    Returns: a pyplot
    '''
    plt.hist(dG_f + np.array(dG_b))
    plt.title('Distribution of fwd-bwd discrepancies')
    plt.xlabel('Difference in delta-G')
    plt.ylabel('Count')
    #return the figure
    return plt.gca()


def plot_general(cumulative,
                 cumulative_ylim,
                 per_window,
                 per_window_ylim,
                 RT,
                 width=8,
                 height=4,
                 pdf_type='KDE',
                 fontsize=12,
                 fig=None,
                 axes=None,
                 hysttype='classic',
                 label=None,
                 color='blue',
                 errorbars=True,
                 xlim=[-1,1]):
    '''
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
    '''
    if fig is None and axes is None:
        fig, axes = plt.subplots(3,2, sharex='col', sharey='row', gridspec_kw={'width_ratios': [2, 1]})
        ((cumul_ax, del1),( each_ax, del2), (hyst_ax, pdf_ax)) = axes

        fig.delaxes(del1)
        fig.delaxes(del2)
    else:
        cumul_ax, each_ax, hyst_ax, pdf_ax = axes

    # Cumulative change in kcal/mol
    cumul_ax.errorbar(cumulative.index, cumulative.BAR.f*RT, yerr=cumulative.BAR.errors, marker=None, linewidth=1, label=label, color=color)
    cumul_ax.set(ylabel=r'Cumulative $\mathrm{\Delta} G_{\lambda}$'+'\n(kcal/mol)', ylim=cumulative_ylim)

    # Per-window change in kcal/mol
    if errorbars:
        each_ax.errorbar(per_window.index, per_window.BAR.df*RT, yerr=per_window.BAR.ddf, marker=None, linewidth=1, color=color)
        each_ax.errorbar(per_window.index, -per_window.EXP.dG_b*RT, marker=None, linewidth=1, alpha=0.5, linestyle='--', color=color)
    each_ax.plot(per_window.index, per_window.EXP.dG_f*RT, marker=None, linewidth=1, alpha=0.5, color=color)

    each_ax.set(ylabel=r'$\mathrm{\Delta} G_\lambda$'+'\n'+r'$\left(kcal/mol\right)$', ylim=per_window_ylim)

    #Hysteresis Plots
    hyst_ax, pdf_ax = plot_hysteresis((hyst_ax, pdf_ax), per_window, hysttype, color, fontsize, pdf_type, xlim)

    fig.set_figwidth(width)
    fig.set_figheight(height*3)
    fig.tight_layout()

    for ax in [cumul_ax,each_ax,hyst_ax,pdf_ax]:
        ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)

    return fig, [cumul_ax,each_ax,hyst_ax,pdf_ax]



def add_hyst_textbox(diff, pdf_x, pdf_y, pdf_ax, xoffset=0.15, yoffset=0.95, fs=14):
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

    textstr = fr"$\rm mode=${np.round(mode,2)}"+"\n"+fr"$\mu$={np.round(mean,2)}"+"\n"+fr"$\sigma$={np.round(std,2)}"
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    pdf_ax.text(xoffset, yoffset, textstr, transform=pdf_ax.transAxes, fontsize=fs,
            verticalalignment='top', bbox=props)
    
    return pdf_ax

def plot_hysteresis(axes,
                    per_window,
                    hysttype,
                    color='blue',
                    fontsize=12,
                    pdf_type='KDE',
                    xlim=[-1,1],
                    textbox=True):
    hyst_ax, pdf_ax = axes
    
    diff = per_window.EXP['difference']
    assert hysttype in ['classic', 'lines'], f"ERROR: I don't know how to plot hysttype={hysttype}"
    if hysttype == 'classic':
        hyst_ax.vlines(per_window.index, np.zeros(len(per_window)), diff, label="fwd - bwd", linewidth=2, color=color)
    elif hysttype == 'lines':
        hyst_ax.plot(per_window.index, diff, label="fwd - bwd", linewidth=1, color=color)

    ytxt = r'$\delta_\lambda$ (kcal/mol)'
    hyst_ax.set_ylabel(ylabel=ytxt, fontsize=fontsize)
    hyst_ax.set_ylim(xlim)

    xtxt = r'$\lambda$'
    hyst_ax.set_xlabel(xlabel=xtxt, fontsize=fontsize)

    if pdf_type=='KDE':
        kernel = sp.stats.gaussian_kde(diff)
        pdf_x = np.linspace(xlim[0], xlim[1], 1000)
        pdf_y = kernel(pdf_x)
        pdf_ax.plot(pdf_y, pdf_x, label='KDE', color=color)
    elif pdf_type=='Histogram':
        pdf_y, pdf_x = np.histogram(diff, density=True)
        pdf_x = pdf_x[:-1]+(pdf_x[1]-pdf_x[0])/2
        pdf_ax.plot(pdf_y, pdf_x,  label="Estimated Distribution", color=color)
    else:
        raise f"Error: PDFtype {pdf_type} not recognized"

    pdf_ax.set_xlabel(pdf_type, fontsize=fontsize)

    if textbox:
        pdf_ax = add_hyst_textbox(diff, pdf_x, pdf_y, pdf_ax)

    return hyst_ax, pdf_ax



def plot_general_legacy(cumulative,
                        cumulative_ylim,
                        perWindow,
                        per_window_ylim,
                        RT,
                        width=8,
                        height=4,
                        PDFtype='KDE',
                        fontsize=12):
    '''
    This is a deprecated plotting function
    '''
    fig, axes = plt.subplots(4,2, sharex='col', sharey='row', gridspec_kw={'width_ratios': [2, 1]})
    ((cumulative_ax, del1),( each_ax, del2), (ddG_ax, del3), (hysteresis_ax, pdf_ax)) = axes

    fig.delaxes(del1)
    fig.delaxes(del2)
    fig.delaxes(del3)

    # Cumulative change in kcal/mol
    cumulative_ax.errorbar(cumulative.index, cumulative.BAR.f*RT, yerr=cumulative.BAR.errors, marker=None, linewidth=1)
    cumulative_ax.set(ylabel=r'Cumulative $\mathrm{\Delta} G_{\lambda}$'+'\n(kcal/mol)', ylim=cumulative_ylim)

    # Per-window change in kcal/mol
    each_ax.errorbar(perWindow.index, perWindow.BAR.df*RT, yerr=perWindow.BAR.ddf, marker=None, linewidth=1)
    each_ax.plot(perWindow.index, perWindow.EXP.dG_f*RT, marker=None, linewidth=1, alpha=0.5)
    each_ax.errorbar(perWindow.index, -perWindow.EXP.dG_b*RT, marker=None, linewidth=1, alpha=0.5)
    each_ax.set(ylabel=r'$\mathrm{\Delta} G_\lambda$'+'\n'+r'$\left(kcal/mol\right)$', ylim=per_window_ylim)

    # Second derivative plot
    ddG = np.diff(perWindow.BAR.df*RT)
    ddG_ax.errorbar(cumulative.index[1:-1], ddG, marker='.')
    #ddGAx.set_xlabel(r'$\lambda$')
    ddG_ax.set_ylabel(r"$\mathrm{\Delta\Delta} G_\lambda \left(kcal/mol\right)$")
    ddG_ax.set(ylim=(-1, 1))

    #Hysteresis Plots
    diff = perWindow.EXP['difference']
    hysteresis_ax.vlines(perWindow.index, np.zeros(len(perWindow)), diff, label="fwd - bwd", linewidth=2)
    hysteresis_ax.set(ylabel=r'$\delta_\lambda$ (kcal/mol)', ylim=(-1,1))
    hysteresis_ax.set_xlabel(xlabel=r'$\lambda$', fontsize=fontsize)

    if PDFtype=='KDE':
        kernel = sp.stats.gaussian_kde(diff)
        pdfx = np.linspace(-1, 1, 1000)
        pdfy = kernel(pdfx)
        pdf_ax.plot(pdfy, pdfx, label='KDE')
    elif PDFtype=='Histogram':
        pdfy, pdfx = np.histogram(diff, density=True)
        pdfx = pdfx[:-1]+(pdfx[1]-pdfx[0])/2
        pdf_ax.plot(pdfy, pdfx,  label="Estimated Distribution")
    else:
        raise(f"Error: PDFtype {PDFtype} not recognized")

    pdf_ax.set_xlabel(PDFtype, fontsize=fontsize)

    std = np.std(diff)
    mean = np.average(diff)
    temp = pd.Series(pdfy, index=pdfx)
    mode = temp.idxmax()

    textstr = r"$\rm mode=$"+f"{np.round(mode,2)}"+"\n"+fr"$\mu$={np.round(mean,2)}"+"\n"+fr"$\sigma$={np.round(std,2)}"
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    pdf_ax.text(0.15, 0.95, textstr, transform=pdf_ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    fig.set_figwidth(width)
    fig.set_figheight(height*3)
    fig.tight_layout()

    for ax in [cumulative_ax,each_ax,hysteresis_ax,pdf_ax, ddG_ax]:
        ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)

    return fig, [cumulative_ax,each_ax,hysteresis_ax,pdf_ax, ddG_ax]





















