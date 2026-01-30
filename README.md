# SAFEP
See detailed documentation here: https://safep-alchemy.readthedocs.io/en/latest/

SAFEP-related scripts and tools

These scripts and notebooks are broadly applicable to FEP analysis.

## Sample Notebooks:
- BAR_Estimator_Basic: General use FEP analysis. Includes cumulative dG estimates, error estimates, and measures of convergence.
- RFEP: For analyzing TI calculations with changing force constants.


## Quickstart:

In a terminal (tested on Linux and Mac):
1. [Optional] Create and activate a conda environment `conda create -n safep` `conda activate safep`
2. Install the latest version of the safep package `pip install git+https://github.com/BranniganLab/safep.git`
3. Continue with either CLI or Notebook interfaces below

### Notebook
4. Open the relevant notebook
  - For FEP results, open BAR_Estimator_Basic.ipynb
  - For RFEP, open RFEP_analysis.ipynb
5. Follow the instructions in the relevant notebook

### CLI (only available for FEP, not yet for RFEP)
4. Navigate to the root directory of your FEP results.

Given a file structure like:
```
|-replica_1
|   |-window1.fepout
|   |-window2.fepout
|   ...
|-replica_2
|   |-window1.fepout
|   |-window2.fepout
|   ...
...
```
5. Run `python -m safep.AFEP_parse --path . --fepoutre "*fepout --replicare "replica_*" --temperature 303.15 --detect_equilibrium True -- make_figures True`

More detailed explanations can be seen by running `python -m safep.AFEP_parse --help`


## Installation:
0. Clone this repository
1. Enter the repository directory
2. Run ``` pip install . ```

OR

0. Run ``` pip install git+https://github.com/BranniganLab/safep.git ```
