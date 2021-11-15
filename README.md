# safep
SAFEP-related scripts and miscellaneous tools

These scripts and notebooks are broadly applicable to FEP analysis.



# Contents:
## AlternativeParser_noBAR:

  - Reads namd fepout files and parses them into a pandas dataframe (O(N) for N samples.
  - Assesses the distribution of dE and the difference between forward and backward samples
  - Can handle incomplete data sets but has minimal error checking. 
  - WILL NOT warn against inconsistencies in fepout files, insufficient sampling, etc. Caveat emptor.
  
## Analysis_Scripts.py:

  - Set of functions to run basic analyses. Runs the analyses in BAR_NAMD_alchemlyb by default
  - Primarily useful for parsing large datasets on distributed compute resources.
  - Saves figures as svgs.
  
  
## BAR_NAMD:

  - Wrapper for alchemlyb analysis.
  - May be adapted to handle partial data sets
  - Runs basic analyses including generating convergence plots and calculating net change with the pymbar MBAR estimator
  - Does not include decorrelation/bootstrapping by default - errors are likely to be under-estimated
