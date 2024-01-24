# Import block
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd

from .helpers import *
from .processing import get_n_samples


def plot_samples(theax, u_nk, color='blue', label='Raw Data'):
    samples = get_n_samples(u_nk)
    theax.plot(samples, color=color, label=label)
    plt.yscale('log') 
    theax.set_ylabel('Number of Samples')
    theax.set_xlabel(r'$\lambda$')
    theax.legend()
    return theax


def convergence_plot(theax, fs, ferr, bs, berr, fwdColor='#0072B2', bwdColor='#D55E00', lgndF=None, lgndB=None):
    '''
    A revised convergence plot. Plays nicely with other plotting functions and is more customizable.
    Arguments: theax[is] on which to plot, fs (forward estimates), ferr (forward errors), bs (backward-sampled estimates), berr (backward-sampled errors), fwdColor, bwdColor, lgndF legend forward color, lgndB legend backward color
    Returns: theax[is]
    '''
    if not lgndF:
        lgndF=fwdColor
        lgndB=bwdColor
        
    theax.errorbar(np.arange(len(fs))/len(fs)+0.1, fs, yerr=ferr, marker='o', linewidth=1, color=fwdColor, markerfacecolor='white', markeredgewidth=1, markeredgecolor=fwdColor, ms=5)
    theax.errorbar(np.arange(len(bs))/len(fs)+0.1, bs, yerr=berr, marker='o', linewidth=1, color=bwdColor, markerfacecolor='white', markeredgewidth=1, markeredgecolor=bwdColor, ms=5, linestyle='--')

    theax.xaxis.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    
    finalMean = fs[-1]
    theax.axhline(y= finalMean, linestyle='-.', color='gray')
    theax.plot(0, finalMean, linewidth=1, color=lgndF, label='Forward Time Sampling')
    theax.plot(0, finalMean, linewidth=1, color=lgndB, linestyle='--', label='Backward Time Sampling')
    
    return theax

def do_conv_plot(ax, X, fs, ferr, fwdColor, label=None):
    '''
    A minimalist convergence plot
    Arguments: ax[is], X (the X values), fs (forward estimates), ferr (forward errors), fwdColor (the forward color), label text
    Returns: ax[is]
    '''
    ax.errorbar(X, fs, yerr=ferr, marker=None, linewidth=1, color=fwdColor, markerfacecolor='white', markeredgewidth=1, markeredgecolor=fwdColor, ms=5, label=label)
    return ax


def convergence_plot(theax, fs, ferr, bs, berr, fwdColor='#0072B2', bwdColor='#D55E00', lgndF=None, lgndB=None, fontsize=12, errorbars=True):
    '''
    Convergence plot. Does the convergence plotting.
    Returns: a pyplot
    '''
    if not lgndF:
        lgndF=fwdColor
        lgndB=bwdColor
        
        
    lower = fs[-1]-ferr[-1]
    upper = fs[-1]+ferr[-1]
    theax.fill_between([0,1],[lower, lower], [upper, upper], color='gray', alpha=0.1)

    fwdparams = {'marker':'o',
                'linewidth':1,
                'color':fwdColor,
                'markerfacecolor':'white',
                'markeredgewidth':1,
                'markeredgecolor':fwdColor,
                'ms':5,
                }
    bwdparams = {'marker':'o',
                'linewidth':1,
                'color':bwdColor,
                'markerfacecolor':'white',
                'markeredgewidth':1,
                'markeredgecolor':bwdColor,
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
    
    finalMean = fs[-1]
    theax.axhline(y= finalMean, linestyle='-.', color='gray')
    theax.set_ylim((finalMean-0.75, finalMean+0.75))
    
    theax.plot(0, finalMean, linewidth=1, color=lgndF, label='Forward Time Sampling')
    theax.plot(0, finalMean, linewidth=1, color=lgndB, linestyle='--', label='Backward Time Sampling')
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
    plt.hist(dG_f + np.array(dG_b));
    plt.title('Distribution of fwd-bwd discrepancies')
    plt.xlabel('Difference in delta-G')
    plt.ylabel('Count')
    #return the figure
    return plt.gca()


def plot_general(cumulative, 
                 cumulativeYlim, 
                 perWindow, 
                 perWindowYlim, 
                 RT, 
                 width=8, 
                 height=4, 
                 PDFtype='KDE', 
                 fontsize=12,
                 fig=None,
                 axes=None,
                 hysttype='classic',
                 label=None,
                 color='blue',
                 errorbars=True):
    if fig is None and axes is None:
        fig, axes = plt.subplots(3,2, sharex='col', sharey='row', gridspec_kw={'width_ratios': [2, 1]})
        ((cumAx, del1),( eachAx, del2), (hystAx, pdfAx)) = axes

        fig.delaxes(del1)
        fig.delaxes(del2)
    else:
        cumAx, eachAx, hystAx, pdfAx = axes

    # Cumulative change in kcal/mol
    cumAx.errorbar(cumulative.index, cumulative.BAR.f*RT, yerr=cumulative.BAR.errors, marker=None, linewidth=1, label=label, color=color)
    cumAx.set(ylabel=r'Cumulative $\mathrm{\Delta} G_{\lambda}$'+'\n(kcal/mol)', ylim=cumulativeYlim)

    # Per-window change in kcal/mol
    if errorbars:
        eachAx.errorbar(perWindow.index, perWindow.BAR.df*RT, yerr=perWindow.BAR.ddf, marker=None, linewidth=1, color=color)
        eachAx.errorbar(perWindow.index, -perWindow.EXP.dG_b*RT, marker=None, linewidth=1, alpha=0.5, linestyle='--', color=color)
    eachAx.plot(perWindow.index, perWindow.EXP.dG_f*RT, marker=None, linewidth=1, alpha=0.5, color=color)
        
    eachAx.set(ylabel=r'$\mathrm{\Delta} G_\lambda$'+'\n'+r'$\left(kcal/mol\right)$', ylim=perWindowYlim)

    #Hysteresis Plots
    diff = perWindow.EXP['difference']
    assert hysttype in ['classic', 'lines'], f"ERROR: I don't know how to plot hysttype={hysttype}"
    if hysttype == 'classic':
        hystAx.vlines(perWindow.index, np.zeros(len(perWindow)), diff, label="fwd - bwd", linewidth=2, color=color)
    elif hysttype == 'lines':
        hystAx.plot(perWindow.index, diff, label="fwd - bwd", linewidth=1, color=color)

    hystAx.set(ylabel=r'$\delta_\lambda$ (kcal/mol)', ylim=(-1,1))
    hystAx.set_xlabel(xlabel=r'$\lambda$', fontsize=fontsize)
    
    if PDFtype=='KDE':
        kernel = sp.stats.gaussian_kde(diff)
        pdfX = np.linspace(-1, 1, 1000)
        pdfY = kernel(pdfX)
        pdfAx.plot(pdfY, pdfX, label='KDE', color=color)
    elif PDFtype=='Histogram':
        pdfY, pdfX = np.histogram(diff, density=True)
        pdfX = pdfX[:-1]+(pdfX[1]-pdfX[0])/2
        pdfAx.plot(pdfY, pdfX,  label="Estimated Distribution", color=color)
    else:
        raise(f"Error: PDFtype {PDFtype} not recognized")
    
    pdfAx.set_xlabel(PDFtype, fontsize=fontsize)

    std = np.std(diff)
    mean = np.average(diff)
    temp = pd.Series(pdfY, index=pdfX)
    mode = temp.idxmax()
    
    textstr = r"$\rm mode=$"+f"{np.round(mode,2)}"+"\n"+fr"$\mu$={np.round(mean,2)}"+"\n"+fr"$\sigma$={np.round(std,2)}"
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    pdfAx.text(0.15, 0.95, textstr, transform=pdfAx.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    fig.set_figwidth(width)
    fig.set_figheight(height*3)
    fig.tight_layout()
    
    for ax in [cumAx,eachAx,hystAx,pdfAx]:
        ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)

    return fig, [cumAx,eachAx,hystAx,pdfAx] 


'''
This is a deprecated plotting function
'''
def plot_general_legacy(cumulative, cumulativeYlim, perWindow, perWindowYlim, RT, width=8, height=4, PDFtype='KDE', fontsize=12):
    fig, axes = plt.subplots(4,2, sharex='col', sharey='row', gridspec_kw={'width_ratios': [2, 1]})
    ((cumAx, del1),( eachAx, del2), (ddGAx, del3), (hystAx, pdfAx)) = axes

    fig.delaxes(del1)
    fig.delaxes(del2)
    fig.delaxes(del3)

    # Cumulative change in kcal/mol
    cumAx.errorbar(cumulative.index, cumulative.BAR.f*RT, yerr=cumulative.BAR.errors, marker=None, linewidth=1)
    cumAx.set(ylabel=r'Cumulative $\mathrm{\Delta} G_{\lambda}$'+'\n(kcal/mol)', ylim=cumulativeYlim)

    # Per-window change in kcal/mol
    eachAx.errorbar(perWindow.index, perWindow.BAR.df*RT, yerr=perWindow.BAR.ddf, marker=None, linewidth=1)
    eachAx.plot(perWindow.index, perWindow.EXP.dG_f*RT, marker=None, linewidth=1, alpha=0.5)
    eachAx.errorbar(perWindow.index, -perWindow.EXP.dG_b*RT, marker=None, linewidth=1, alpha=0.5)
    eachAx.set(ylabel=r'$\mathrm{\Delta} G_\lambda$'+'\n'+r'$\left(kcal/mol\right)$', ylim=perWindowYlim)

    # Second derivative plot
    ddG = np.diff(perWindow.BAR.df*RT)
    ddGAx.errorbar(cumulative.index[1:-1], ddG, marker='.')
    #ddGAx.set_xlabel(r'$\lambda$')
    ddGAx.set_ylabel(r"$\mathrm{\Delta\Delta} G_\lambda \left(kcal/mol\right)$")
    ddGAx.set(ylim=(-1, 1))

    #Hysteresis Plots
    diff = perWindow.EXP['difference']
    hystAx.vlines(perWindow.index, np.zeros(len(perWindow)), diff, label="fwd - bwd", linewidth=2)
    hystAx.set(ylabel=r'$\delta_\lambda$ (kcal/mol)', ylim=(-1,1))
    hystAx.set_xlabel(xlabel=r'$\lambda$', fontsize=fontsize)
    
    if PDFtype=='KDE':
        kernel = sp.stats.gaussian_kde(diff)
        pdfX = np.linspace(-1, 1, 1000)
        pdfY = kernel(pdfX)
        pdfAx.plot(pdfY, pdfX, label='KDE')
    elif PDFtype=='Histogram':
        pdfY, pdfX = np.histogram(diff, density=True)
        pdfX = pdfX[:-1]+(pdfX[1]-pdfX[0])/2
        pdfAx.plot(pdfY, pdfX,  label="Estimated Distribution")
    else:
        raise(f"Error: PDFtype {PDFtype} not recognized")
    
    pdfAx.set_xlabel(PDFtype, fontsize=fontsize)

    std = np.std(diff)
    mean = np.average(diff)
    temp = pd.Series(pdfY, index=pdfX)
    mode = temp.idxmax()
    
    textstr = r"$\rm mode=$"+f"{np.round(mode,2)}"+"\n"+fr"$\mu$={np.round(mean,2)}"+"\n"+fr"$\sigma$={np.round(std,2)}"
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    pdfAx.text(0.15, 0.95, textstr, transform=pdfAx.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    fig.set_figwidth(width)
    fig.set_figheight(height*3)
    fig.tight_layout()
    
    for ax in [cumAx,eachAx,hystAx,pdfAx, ddGAx]:
        ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)

    return fig, [cumAx,eachAx,hystAx,pdfAx, ddGAx] 





















