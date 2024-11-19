#Large datasets can be difficult to parse on a workstation due to inefficiencies in the way data is represented for pymbar. When possible, reduce the size of your dataset.
import matplotlib.pyplot as plt

from glob import glob #file regexes
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm #for progress bars
import re #regex
from natsort import natsorted #for sorting "naturally" instead of alphabetically

from alchemlyb.visualisation.dF_state import plot_dF_state

from alchemlyb.parsing import namd
from alchemlyb.estimators import BAR
from alchemlyb.visualisation.dF_state import plot_dF_state
from alchemlyb.visualisation import plot_convergence
from safep.processing import read_and_process

import re


import safep
import alchemlyb
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from alchemlyb.parsing import namd
from IPython.display import display, Markdown
from pathlib import Path
from dataclasses import dataclass
import scipy as sp
from alchemlyb.estimators import BAR
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
import pandas as pd
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--path', type=str, help='The absolute path to the directory containing the fepout files', default='.')
    parser.add_argument('--fepoutre', type=str, help='A regular expression that matches the fepout files of interest.', default='*.fep*')
    parser.add_argument('--replicare', type=str, help='A regular expression that matches the replica directories', default='Replica?')
    parser.add_argument('--temperature', type=float, help='The temperature at which the FEP was run.')
    parser.add_argument('--decorrelate', type=bool, help='Flag to determine whether or not to decorrelate the data. (1=decorrelate, 0=use all data)', default=0)
    parser.add_argument('--detectEQ', type=bool, help='Flag for automated equilibrium detection.', default=0)
    parser.add_argument('--fittingMethod', type=str, help='Method for fitting the forward-backward discrepancies (hysteresis). LS=least squares, ML=maximum likelihood Default: LS', default='LS')
    parser.add_argument('--maxSize', type=float, help='Maximum total file size in GB. This is MUCH less than the required RAM. Default: 1', default=1)
    parser.add_argument('--makeFigures', type=bool, help='Run additional diagnostics and save figures to the directory. default: False', default=0)

    args = parser.parse_args()

    dataroot = Path('.')
    replica_pattern=args.replicare
    replicas = dataroot.glob(replica_pattern)
    filename_pattern=args.fepoutre

    temperature = 303.15
    RT = 0.00198720650096 * temperature
    detectEQ = True #Flag for automated equilibrium detection


    colors = ['blue', 'red', 'green', 'purple', 'orange', 'violet', 'cyan']
    itcolors = iter(colors)

    @dataclass
    class FepRun:
        u_nk:           pd.DataFrame
        perWindow:      pd.DataFrame
        cumulative:     pd.DataFrame
        forward:        pd.DataFrame
        forward_error:  pd.DataFrame
        backward:       pd.DataFrame
        backward_error: pd.DataFrame
        per_lambda_convergence: pd.DataFrame
        color: str

    # # Extract key features from the MBAR fitting and get ΔG
    # Note: alchemlyb operates in units of kT by default. We multiply by RT to convert to units of kcal/mol.

    # # Read and plot number of samples after detecting EQ


    fepruns = {}
    for replica in replicas:
        print(f"Reading {replica}")
        unkpath = replica.joinpath('decorrelated.csv')
        u_nk = None
        if unkpath.is_file():
            print(f"Found existing dataframe. Reading.")
            u_nk = safep.read_UNK(unkpath)
        else:
            print(f"Didn't find existing dataframe at {unkpath}. Checking for raw fepout files.")
            fepoutFiles = list(replica.glob(filename_pattern))
            totalSize = 0
            for file in fepoutFiles:
                totalSize += os.path.getsize(file)
            print(f"Will process {len(fepoutFiles)} fepout files.\nTotal size:{np.round(totalSize/10**9, 2)}GB")

            if len(list(fepoutFiles))>0:
                print("Reading fepout files")
                fig, ax = plt.subplots()

                u_nk = namd.extract_u_nk(fepoutFiles, temperature)
                u_nk = u_nk.sort_index(axis=0, level=1).sort_index(axis=1)
                safep.plot_samples(ax, u_nk, color='blue', label='Raw Data')

                if detectEQ:
                    print("Detecting equilibrium")
                    u_nk = safep.detect_equilibrium_u_nk(u_nk)
                    safep.plot_samples(ax, u_nk, color='orange', label='Equilibrium-Detected')

                plt.savefig(f"./{str(replica)}_FEP_number_of_samples.pdf")
                plt.show()
                safep.save_UNK(u_nk, unkpath)
            else:
                print(f"WARNING: no fepout files found for {replica}. Skipping.")
        
        if u_nk is not None:
            fepruns[str(replica)] = FepRun(u_nk, None, None, None, None, None, None, None, next(itcolors))
            


    # In[ ]:


    for key, feprun in fepruns.items():
        u_nk = feprun.u_nk
        feprun.perWindow, feprun.cumulative = safep.do_estimation(u_nk) #Run the BAR estimator on the fep data
        feprun.forward, feprun.forward_error, feprun.backward, feprun.backward_error = safep.do_convergence(u_nk) #Used later in the convergence plot'
        feprun.per_lambda_convergence = safep.do_per_lambda_convergence(u_nk)



    # # Plot data

    # In[ ]:


    toprint = ""
    dGs = []
    errors = []
    for key, feprun in fepruns.items():
        cumulative = feprun.cumulative
        dG = np.round(cumulative.BAR.f.iloc[-1]*RT, 1)
        error = np.round(cumulative.BAR.errors.iloc[-1]*RT, 1)
        dGs.append(dG)
        errors.append(error)

        changeAndError = f'{key}: \u0394G = {dG}\u00B1{error} kcal/mol'
        toprint += '<font size=5>{}</font><br/>'.format(changeAndError)

    toprint += '<font size=5>{}</font><br/>'.format('__________________')
    mean = np.average(dGs)

    #If there are only a few replicas, the MBAR estimated error will be more reliable, albeit underestimated
    if len(dGs)<3:
        sterr = np.sqrt(np.sum(np.square(errors)))
    else:
        sterr = np.round(np.std(dGs),1)
    toprint += '<font size=5>{}</font><br/>'.format(f'mean: {mean}')
    toprint += '<font size=5>{}</font><br/>'.format(f'sterr: {sterr}')
    Markdown(toprint)



    def do_agg_data(dataax, plotax):
        agg_data = []
        lines = dataax.lines
        for line in lines:
            agg_data.append(line.get_ydata())
        flat = np.array(agg_data).flatten()
        kernel = sp.stats.gaussian_kde(flat)
        pdfX = np.linspace(-1, 1, 1000)
        pdfY = kernel(pdfX)
        std = np.std(flat)
        mean = np.average(flat)
        temp = pd.Series(pdfY, index=pdfX)
        mode = temp.idxmax()

        textstr = r"$\rm mode=$"+f"{np.round(mode,2)}"+"\n"+fr"$\mu$={np.round(mean,2)}"+"\n"+fr"$\sigma$={np.round(std,2)}"
        props = dict(boxstyle='square', facecolor='white', alpha=1)
        plotax.text(0.175, 0.95, textstr, transform=plotax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

        return plotax


    fig = None
    for key, feprun in fepruns.items():
        if fig is None:
            fig, axes = safep.plot_general(feprun.cumulative, None, feprun.perWindow, None, RT, hysttype='lines', label=key, color=feprun.color)
            axes[1].legend()
        else:
            fig, axes = safep.plot_general(feprun.cumulative, None, feprun.perWindow, None, RT, fig=fig, axes=axes, hysttype='lines', label=key, color=feprun.color)
        #fig.suptitle(changeAndError)

    # hack to get aggregate data:
    axes[3] = do_agg_data(axes[2], axes[3])

    axes[0].set_title(str(mean)+r'$\pm$'+str(sterr)+' kcal/mol')
    axes[0].legend()
    plt.savefig(dataroot.joinpath('FEP_general_figures.pdf'))


    # # Plot the estimated total change in free energy as a function of simulation time; contiguous subsets starting at t=0 ("Forward") and t=end ("Reverse")


    fig, convAx = plt.subplots(1,1)

    for key, feprun in fepruns.items():
        convAx = safep.convergence_plot(convAx, 
                                        feprun.forward*RT, 
                                        feprun.forward_error*RT, 
                                        feprun.backward*RT,
                                        feprun.backward_error*RT,
                                        fwd_color=feprun.color,
                                        bwd_color=feprun.color,
                                        errorbars=False
                                        )
        convAx.get_legend().remove()

    forward_line, = convAx.plot([],[],linestyle='-', color='black', label='Forward Time Sampling')
    backward_line, = convAx.plot([],[],linestyle='--', color='black', label='Backward Time Sampling')
    convAx.legend(handles=[forward_line, backward_line])
    ymin = np.min(dGs)-1
    ymax = np.max(dGs)+1
    convAx.set_ylim((ymin,ymax))
    plt.savefig(dataroot.joinpath('FEP_convergence.pdf'))

    genfig = None
    for key, feprun in fepruns.items():
        if genfig is None:
            genfig, genaxes = safep.plot_general(feprun.cumulative, None, feprun.perWindow, None, RT, hysttype='lines', label=key, color=feprun.color)
        else:
            genfig, genaxes = safep.plot_general(feprun.cumulative, None, feprun.perWindow, None, RT, fig=genfig, axes=genaxes, hysttype='lines', label=key, color=feprun.color)
    plt.delaxes(genaxes[0])
    plt.delaxes(genaxes[1])

    genaxes[3] = do_agg_data(axes[2], axes[3])
    genaxes[2].set_title(str(mean)+r'$\pm$'+str(sterr)+' kcal/mol')

    for txt in genfig.texts:
        print(1)
        txt.set_visible(False)
        txt.set_text("")
    plt.show()
    plt.savefig(dataroot.joinpath('FEP_perLambda_convergence.pdf'))




    if args.makeFigures == 1:
        # Cumulative change in kT
        plt.errorbar(l, f, yerr=errors, marker='.')
        plt.xlabel('lambda')
        plt.ylabel('DeltaG(lambda) (kT)')
        plt.title(f'Cumulative dG with accumulated errors {affix}\n{changeAndError}')
        plt.savefig(f'{path}dG_cumulative_kT_{affix}.png', dpi=600)
        plt.clf()

        # Per-window change in kT
        plt.errorbar(l_mid, df, yerr=ddf, marker='.')
        plt.xlabel('lambda')
        plt.ylabel('Delta G per window (kT)')
        plt.title(f'Per-Window dG with individual errors {affix}')
        plt.savefig(f'{path}dG_{affix}.png', dpi=600)
        plt.clf()

        # Per-window change in kT
        plt.errorbar(l[1:-1], np.diff(df), marker='.')
        plt.xlabel('lambda (L)')
        plt.ylabel("dG'(L)")
        plt.title(f'derivative of dG {affix}')
        plt.savefig(f'{path}dG_prime_{affix}.png', dpi=600)
        plt.clf()

        ####
        try:
            convergence_plot(u_nk, l)
            plt.title(f'Convergence {affix}')
            plt.savefig(f'{path}convergence_{affix}.png', dpi=600)
            plt.clf()
        except:
            print("Failed to generate convergence plot. Probably due to too few samples after decorrelation.")

        ####
        l, l_mid, dG_f, dG_b = get_EXP(u_nk)
        plt.vlines(l_mid, np.zeros(len(l_mid)), dG_f + np.array(dG_b), label="fwd - bwd", linewidth=2)

        plt.legend()
        plt.title(f'Fwd-bwd discrepancies by lambda {affix}')
        plt.xlabel('Lambda')
        plt.ylabel('Diff. in delta-G')
        plt.savefig(f'{path}discrepancies_{affix}.png', dpi=600)
        plt.clf()


        ###
        #Do residual fitting
        ###
        X, Y, pdfX, pdfY, fitted, pdfXnorm, pdfYnorm, pdfYexpected = getPDF(dG_f, dG_b)
        
        #plot the data
        fig, (pdfAx, pdfResid) = plt.subplots(2, 1, sharex=True)
        plt.xlabel('Difference in delta-G')
        
        pdfAx.plot(pdfX, pdfY,  label="Estimated Distribution")
        pdfAx.set_ylabel("PDF")
        pdfAx.plot(pdfXnorm, pdfYnorm, label="Fitted Normal Distribution", color="orange")

        #pdf residuals
        pdfResiduals = pdfY-pdfYexpected
        pdfResid.plot(pdfX, pdfResiduals)
        pdfResid.set_ylabel("PDF residuals") 

        fig.set_figheight(10)
        if DiscrepancyFitting == 'LS':
            pdfAx.title.set_text(f"Least squares fitting of cdf(fwd-bkwd)\nSkewness: {np.round(skew(X),2)}\nFitted parameters: Mean={np.round(fitted[0],3)}, Stdv={np.round(fitted[1],3)}\nPopulation parameters: Mean={np.round(np.average(X),3)}, Stdv={np.round(np.std(X),3)}")
            plt.savefig(f"{path}LeastSquares_pdf_{affix}.png", dpi=600)
        elif DiscrepancyFitting == 'ML':
            pdfAx.title.set_text(f"Maximum likelihood fitting of fwd-bkwd\nFitted parameters: Mean={np.round(fitted[0],3)}, Stdv={np.round(fitted[1],3)}\nPopulation parameters: Mean={np.round(np.average(X),3)}, Stdv={np.round(np.std(X),3)}")
            plt.savefig(f"{path}MaximumLikelihood_pdf_{affix}.png", dpi=600)
        plt.clf()

