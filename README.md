# SAFEP
See detailed documentation here: https://safep-alchemy.readthedocs.io/en/latest/

SAFEP-related scripts and tools

These scripts and notebooks are broadly applicable to FEP analysis.

## Sample Notebooks:
- BAR_Estimator_Basic: General use FEP analysis. Includes cumulative dG estimates, error estimates, and measures of convergence.
- RFEP: For analyzing TI calculations with changing force constants.


## Quickstart:

### Installation
In a terminal (tested on Linux and Mac):
0. [Optional] Create and activate a conda environment `conda create -n safep` `conda activate safep`
**Option 1 (includes sample notebooks):**
1. Clone this repository using 
2. Enter the repository directory
3. Run ``` pip install . ```

**Option 2 (only provides CLI):**
1. Run ``` pip install git+https://github.com/BranniganLab/safep.git ```

### Running in a Jupyter Notebook
1. Download one or more example notebooks from the GitHub: https://github.com/BranniganLab/safep/tree/main/Sample_Notebooks
2. [Optional] or clone the repository with `git clone https://github.com/BranniganLab/safep.git`
3. Open the relevant notebook
  - For FEP results, open BAR_Estimator_Basic.ipynb
  - For RFEP/thermodynamic integration, open RFEP_analysis.ipynb
4. Follow the instructions in the notebook

### CLI (only available for FEP, not yet for RFEP)
1. Navigate to the root directory of your FEP results.

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
2. Run `python -m safep.AFEP_parse --path . --fepoutre "*fepout --replicare "replica_*" --temperature 303.15 --detect_equilibrium True -- make_figures True`

More detailed explanations can be seen by running `python -m safep.AFEP_parse --help`



