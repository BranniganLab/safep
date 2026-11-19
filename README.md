# SAFEP
See detailed documentation here: https://safep-alchemy.readthedocs.io/en/latest/

SAFEP-related scripts and tools

These scripts and notebooks are broadly applicable to FEP analysis.

## Sample Notebooks
- BAR_Estimator_Basic: General use FEP analysis. Includes cumulative dG estimates, error estimates, and measures of convergence.
- RFEP_analysis: For analyzing TI calculations with changing force constants.

Older notebooks (AlternativeParser_noBAR, BAR_Estimator_Expanded, Batch_Basic) are still available in `Sample_Notebooks/Legacy/`.


## Quickstart

### Installation
In a terminal (tested on Linux and Mac):

**[Optional] Create and activate a virtual environment**

   Use one of the following (safep requires Python >= 3.10; the RFEP notebook currently needs >= 3.12):
   - venv: `python3 -m venv safep_venv; source safep_venv/bin/activate` (check `python3 --version` first)
   - conda: `conda create -n safep "python>=3.12" pip; conda activate safep`

**Installation Option 1: from a clone (also gives you the sample notebooks and sample data locally):**
1. Clone this repository using `git clone https://github.com/BranniganLab/safep.git`
2. Enter the repository directory `cd safep`
3. Run `pip install .`

**Installation Option 2: direct pip install (library and CLI; get the notebooks separately, see below):**
1. Run `pip install git+https://github.com/BranniganLab/safep.git`

### Running in a Jupyter Notebook
1. Install Jupyter in the same environment: `pip install jupyterlab`
2. Get the sample notebooks:
   - If you used Option 1, they are in `safep/Sample_Notebooks/`, together with runnable sample data (`Replica1`-`Replica3`).
   - Otherwise, download them from https://github.com/BranniganLab/safep/tree/main/Sample_Notebooks (open the notebook, then use the "Download raw file" button).
3. Put the notebook in the directory that contains your data, or edit the paths in its first cells:
   - For FEP results, open `BAR_Estimator_Basic.ipynb` (expects `Replica?/*.fepout` by default)
   - For RFEP/thermodynamic integration, open `RFEP_analysis.ipynb` (expects a Colvars log file, `RFEP_decouple.log` by default)
4. Start Jupyter with `jupyter lab` and follow the instructions in the notebook.

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
2. Run `python -m safep.AFEP_parse --path . --fepoutre "*fepout" --replicare "replica_*" --temperature 303.15 --detect_equilibrium --make_figures`

More detailed explanations can be seen by running `python -m safep.AFEP_parse --help`



