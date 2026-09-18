"""Organize all the data associated with a FEP replica"""

import os
import warnings
from typing import get_args, get_type_hints
from pathlib import Path
from dataclasses import dataclass, fields, field

import numpy as np
import pandas as pd
from alchemlyb.parsing import namd
from matplotlib import pyplot as plt
from safep.moving_wall_TI import ColvarsTraj

import safep


def colvars_path_for_fepout(fepout_file: Path) -> Path:
    """Return the exact same-basename colvars trajectory for a fepout file."""
    fepout_file = Path(fepout_file)
    if fepout_file.suffix != ".fepout":
        raise ValueError(f"Expected a .fepout file, got {fepout_file}")
    return fepout_file.with_suffix(".colvars.traj")


def filter_u_nk_by_dbc(
    u_nk: pd.DataFrame,
    dbc: pd.Series,
    fepout_file: Path | str,
    dbc_min: float | None,
    dbc_max: float | None,
) -> pd.DataFrame:
    """Align energy samples to DBC values and apply optional inclusive bounds."""
    times = u_nk.index.get_level_values("time")
    # Discard trajectory-only rows before checking duplicate values.
    relevant_dbc = dbc[dbc.index.isin(times)]
    duplicates = relevant_dbc[relevant_dbc.index.duplicated(keep=False)]
    for step, values in duplicates.groupby(level=0):
        values = values.dropna()
        if len(values) > 1 and not np.allclose(values, values.iloc[0]):
            raise ValueError(f"Conflicting DBC values found for timestep {step}")
    relevant_dbc = relevant_dbc.groupby(level=0).first()
    aligned_dbc = relevant_dbc.reindex(times)
    missing = aligned_dbc.isna()
    missing_count = int(missing.sum())
    if missing_count:
        warnings.warn(
            f"Discarding {missing_count} fepout samples from {fepout_file} "
            "because DBC data is missing.",
            UserWarning,
            stacklevel=2,
        )

    keep = ~missing
    if dbc_min is not None:
        keep &= aligned_dbc >= dbc_min
    if dbc_max is not None:
        keep &= aligned_dbc <= dbc_max

    result = u_nk.loc[np.asarray(keep, dtype=bool)]
    if result.empty:
        raise ValueError(f"DBC filtering left no samples for {fepout_file}")
    return result


def load_dbc_for_fepouts(fepout_files: list[Path]) -> pd.Series:
    """Load matching DBC trajectories and reconcile their timestep index."""
    trajectories = []
    for fepout_file in fepout_files:
        colvars_file = colvars_path_for_fepout(fepout_file)
        if not colvars_file.is_file():
            raise FileNotFoundError(
                f"No colvars trajectory found for {fepout_file}; expected {colvars_file}"
            )
        trajectory = ColvarsTraj.read_colvars_traj(colvars_file)
        if "DBC" not in trajectory.columns:
            raise ValueError(f"Colvars trajectory {colvars_file} has no DBC column")
        trajectories.append(trajectory["DBC"])

    return pd.concat(trajectories)


def has_dbc_filter(args) -> bool:
    return (
        getattr(args, "dbc_min", None) is not None
        or getattr(args, "dbc_max", None) is not None
    )


def dbc_bounds(args) -> tuple[float | None, float | None]:
    """Return the configured DBC bounds from an AFEP argument object."""
    return getattr(args, "dbc_min", None), getattr(args, "dbc_max", None)


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
        if unkpath.is_file() and not has_dbc_filter(args):
            print("Found existing dataframe. Reading.")
            u_nk = safep.read_UNK(unkpath)
        else:
            if has_dbc_filter(args) and unkpath.is_file():
                print("DBC filtering enabled; bypassing existing dataframe cache.")
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
            if has_dbc_filter(args):
                dbc = load_dbc_for_fepouts(fepout_files)
                dbc_min, dbc_max = dbc_bounds(args)
                u_nk = filter_u_nk_by_dbc(u_nk, dbc, replica, dbc_min, dbc_max)
            fepruns[str(replica)] = FepRun(u_nk, None, None, None, None, None, None, None,
                                           next(itcolors))
    return fepruns

_COERCERS = {
    pd.DataFrame: pd.DataFrame,
    np.ndarray: lambda v: np.asarray(v).flatten(),
    str: str,
}

@dataclass
class FepRun:
    """Datastructure for holding FEP replicas including:
        energies (u_nk)
        free energies
        and associated metrics"""
    u_nk: pd.DataFrame = field(metadata={"skip_sanitize": True})
    per_window: pd.DataFrame|None = None
    cumulative: pd.DataFrame|None = None
    forward: np.ndarray|None = None
    forward_error: np.ndarray|None = None
    backward: np.ndarray|None = None
    backward_error: np.ndarray|None = None
    per_lambda_convergence: pd.DataFrame|None = None
    color: str|None = "k"

    def __post_init__(self):
        self.u_nk.columns = self.u_nk.columns.astype(float)

        # Run the BAR estimator on the fep data
        if self.per_window is None or self.cumulative is None:
            self.per_window, self.cumulative = safep.do_estimation(self.u_nk)

        if (self.forward is None or
                self.forward_error is None or
                self.backward is None or
                self.backward_error is None):
            (
                self.forward,
                self.forward_error,
                self.backward,
                self.backward_error,
            ) = safep.do_convergence(self.u_nk)  # Used later in the convergence plot

        if self.per_lambda_convergence is None:
            self.per_lambda_convergence = safep.do_per_lambda_convergence(self.u_nk)

        self._sanitize_attributes()

    def _sanitize_attributes(self):
        """Coerce each attribute to be consistent with the type hints.

        Args:
            None

        Returns:
            None

        Side Effects:
            After running, all attributes are of the correct type.
        """
        hints = get_type_hints(type(self))
        for fld in fields(self):
            if fld.metadata.get("skip_sanitize"):
                continue

            attr = getattr(self, fld.name)
            hint = hints[fld.name]
            args = get_args(hint)  # unwrap Optional[X]/Union[X, ...]
            main_type = args[0] if args else hint

            if isinstance(attr, main_type) or attr is None:
                continue

            coerce = _COERCERS.get(main_type)
            if coerce is None:
                raise TypeError(
                    f"{type(self).__name__}.{fld.name} can't process type {hint}"
                )
            setattr(self, fld.name, coerce(attr))

    def to_dir(self, root: Path):
        """Write FepRun to a directory

        Arguments:
            root (Path): the directory to write to

        Returns:
            None

        Side Effects:
            Creates and populates the root directory with the fields of a FepRun
        """
        root.mkdir(parents=True, exist_ok=True)
        for field in fields(self):
            attr = getattr(self, field.name)
            if isinstance(attr, np.ndarray):
                attr = pd.DataFrame(attr)
            if isinstance(attr, pd.DataFrame):
                attr.to_csv(root/f'{field.name}.csv')
            else:
                with open(root/f'{field.name}.txt', 'w', encoding="UTF8") as f:
                    f.write(attr)

    @classmethod
    def from_dir(cls, root: Path):
        """Read FepRun from directory

        Arguments:
            root (Path): directory to read

        Returns:
            FepRun: populated from csvs and text files in the directory
        """
        nacent_dict = {}
        key = "u_nk"
        nacent_dict[key] = pd.read_csv(root/f"{key}.csv", header=[0], index_col=[0,1], dtype=float)

        for key in ["per_window", "cumulative"]:
            fname = root / f"{key}.csv"
            nacent_dict[key] = pd.read_csv(fname, header=[0, 1], index_col=[0], dtype=float)

        for key in ["forward", "forward_error", "backward", "backward_error"]:
            fname = root / f"{key}.csv"
            nacent_dict[key] = pd.read_csv(fname, usecols=[1], dtype=float)
            nacent_dict[key].columns = nacent_dict[key].columns.astype(int)

        key = "per_lambda_convergence"
        fname = root / f"{key}.csv"
        nacent_dict[key] = pd.read_csv(fname, header=[0, 1], index_col=[0], dtype=float)

        with open(root/"color.txt", 'r', encoding="UTF8") as f:
            color = f.read()

        return cls(color = color, **nacent_dict)


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
        safep.save_UNK(u_nk, unkpath) # Only save decorrelated samples
    fig.savefig(args.dataroot.joinpath(f"{str(replica)}_FEP_number_of_samples.pdf"))
    return u_nk
