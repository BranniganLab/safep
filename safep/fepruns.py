"""Organize all the data associated with a FEP replica"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from alchemlyb.parsing import namd
from matplotlib import pyplot as plt

import safep


def process_replicas(args, itcolors):
    """Populate a dictionary of Fepruns based on information from a AFEPArguments object

    Args:
        args (AFEPArguments): information needed to collect fep data and analyze it
        itcolors (iterator): an iterator over a list of matplotlib-compatible colors

    Returns:
        dict[FepRun]: a dictionary of all replicas
    """
    # # Extract key features from the MBAR fitting and get ΔG
    # Note: alchemlyb operates in units of kT by default.
    # We multiply by RT to convert to units of kcal/mol.
    fepruns = {}
    root = args.dataroot
    for replica in args.replicas:
        print(f"Reading {replica}")
        unkpath = root/replica/"decorrelated.csv"
        u_nk = None
        if unkpath.is_file():
            print("Found existing dataframe. Reading.")
            u_nk = safep.read_UNK(unkpath)
        else:
            print(
                f"Didn't find existing dataframe at {unkpath}. Checking for raw fepout files.")
            fepout_files = list((root/replica).glob(args.filename_pattern))
            report_number_and_size_of_fepout_files(fepout_files)

            if len(list(fepout_files)) > 0:
                print("Reading fepout files")
                u_nk = read_and_decorrelate(
                    args, replica, unkpath, fepout_files)
            else:
                print(
                    f"WARNING: no fepout files found for {replica}. Skipping.")

        if u_nk is not None:
            fepruns[str(replica)] = FepRun(u_nk, None, None, None, None, None, None, None,
                                           next(itcolors))
    return fepruns


@dataclass
class FepRun:
    """Datastructure for holding FEP replicas including:
        energies (u_nk)
        free energies
        and associated metrics"""
    u_nk: pd.DataFrame
    per_window: pd.DataFrame
    cumulative: pd.DataFrame
    forward: pd.DataFrame
    forward_error: pd.DataFrame
    backward: pd.DataFrame
    backward_error: pd.DataFrame
    per_lambda_convergence: pd.DataFrame
    color: str

    def __post_init__(self):
        # Run the BAR estimator on the fep data
        self.per_window, self.cumulative = safep.do_estimation(self.u_nk)
        (
            self.forward,
            self.forward_error,
            self.backward,
            self.backward_error,
        ) = safep.do_convergence(self.u_nk)  # Used later in the convergence plot
        self.per_lambda_convergence = safep.do_per_lambda_convergence(
            self.u_nk)


def report_number_and_size_of_fepout_files(fepout_files):
    """Check the number and size of all fepout files.

    Proxy for checking if the data will fit in RAM.

    Side effects:
        print stats to stout
    """
    total_size = 0
    for file in fepout_files:
        total_size += os.path.getsize(file)
    print(f"Will process {len(fepout_files)} fepout files." +
          f"\nTotal size:{np.round(total_size/10**9, 2)}GB")


def read_and_decorrelate(args, replica, unkpath, fepout_files):
    """Read each replica and optionally detect equilibrium/decorrelate samples

    Args:
        args (AFEPArguments): information needed to collect fep data and analyze
        replica (str): name of the replica
        unkpath (pathlib.Path): path to save the postprocessed data
        fepout_files (list): list of paths to fepout files

    Returns:
        pd.DataFrame: a pandas DataFrame containing the postprocessed energies
    """
    fig, ax = plt.subplots()

    u_nk = namd.extract_u_nk(fepout_files, args.temperature)
    u_nk = u_nk.sort_index(axis=0, level=1).sort_index(axis=1)
    safep.plot_samples(ax, u_nk, color="blue", label="Raw Data")

    if args.detect_equilibrium:
        print("Detecting equilibrium")
        u_nk = safep.detect_equilibrium_u_nk(u_nk)
        safep.plot_samples(ax, u_nk, color="orange",
                           label="Equilibrium-Detected")
    fig.savefig(args.dataroot.joinpath(f"{str(replica)}_FEP_number_of_samples.pdf"))
    safep.save_UNK(u_nk, unkpath)
    return u_nk
