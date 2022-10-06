#Large datasets can be difficult to parse on a workstation due to inefficiencies in the way data is represented for pymbar. When possible, reduce the size of your dataset.
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import linregress as lr
from scipy.stats import norm

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

import re




if __name__ == '__main__':

    import argparse
    import os
  
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--path', type=str, help='The absolute path to the directory containing the fepout files', default='.')
    parser.add_argument('--fepoutre', type=str, help='A regular expression that matches the fepout files of interest.', default='*.fep*')
    parser.add_argument('--temperature', type=float, help='The temperature at which the FEP was run.')
    parser.add_argument('--decorrelate', type=bool, help='Flag to determine whether or not to decorrelate the data. (1=decorrelate, 0=use all data)', default=0)
    parser.add_argument('--detectEQ', type=bool, help='Flag for automated equilibrium detection.', default=0)
    parser.add_argument('--fittingMethod', type=str, help='Method for fitting the forward-backward discrepancies (hysteresis). LS=least squares, ML=maximum likelihood Default: LS', default='LS')
    parser.add_argument('--maxSize', type=float, help='Maximum total file size in GB. This is MUCH less than the required RAM. Default: 1', default=1)
    parser.add_argument('--makeFigures', type=bool, help='Run additional diagnostics and save figures to the directory. default: False', default=0)

    args = parser.parse_args()

    path = args.path
    filename = args.fepoutre
    maxSize = args.maxSize
    temperature = args.temperature
    decorrelate = (args.decorrelate==1)
    detectEQ = (args.detectEQ==1)
    DiscrepancyFitting = args.fittingMethod
    
    RT = 0.00198720650096 * temperature # ca. 0.59kcal/mol


    fepoutFiles = glob(path+filename)
    print(f"Processing: {path+filename}")

    totalSize = 0
    for file in fepoutFiles:
        totalSize += os.path.getsize(file)
    print(f"Fepout Files: {len(fepoutFiles)}\n")

    if (totalSize/10**9)>maxSize:
        print(f"Error: desired targets (Total size:{np.round(totalSize/10**9, 2)}GB) exceed your max size. Either increase your maximum acceptable file size, or use the 'Extended' notebook")
        raise


    print(f'DetectEQ: {detectEQ}')
    print(f'Decorr: {decorrelate}')
    u_nk, affix = readAndProcess(fepoutFiles, temperature, decorrelate, detectEQ)


    u_nk = u_nk.sort_index(level=1)
    bar = BAR()
    bar.fit(u_nk)
    l, l_mid, f, df, ddf, errors = get_BAR(bar)
    changeAndError = f'\u0394G = {np.round(f.iloc[-1]*RT, 1)}\u00B1{np.round(errors[-1], 3)} kcal/mol'
    print(changeAndError)

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

