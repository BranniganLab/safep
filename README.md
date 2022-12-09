# SAFEP
SAFEP-related scripts and tools

These scripts and notebooks are broadly applicable to FEP analysis.

## Sample Notebooks:
-  AlternativeParser_nBAR uses less RAM to read large fepout files. Rarely useful unless each fepout file is extremely large.
- BAR_Estimator_Basic: General use FEP analysis. Includes cumulative dG estimates, error estimates, and measures of convergence.
- BAR_Estimator_Expanded: Extendend functionality including bootstrapped error estimates, kernel estimation of dL distributions, and more.
- Batch_Basic: Like BAR_Estimator_Basic but designed for multiple replicas of the same calculation.


## Installation:
0. Clone this repository
1. Enter the repository directory
2. Run ``` pip install . ```

OR

0. Run ``` pip install git+https://github.com/BranniganLab/safep.git ```
