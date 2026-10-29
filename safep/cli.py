import argparse
from dataclasses import dataclass
from pathlib import Path

from natsort import natsorted
from scipy.constants import R, calorie


class AFEPArgumentParser(argparse.ArgumentParser):
    """Dedicated CLI argument parser for AFEP.

    Attributes:
        path (str|Path): root path for data folder.
        fepoutre (str): regex for fepout files
        replicare (str): regex for replica directories
        temperature (float): the temperature at which the simulation was run (K)
        detect_equilibrium (bool): Flag to run automated equilibrium detection and downsampling
        fittingMethod (str): Method for fitting forward-backward discrepancies. Untested.
        max_size (float): UNUSED. Maximum size of data to parse. Default 1GB
            Note: this should be much less than the total RAM available.
        make_figures (bool): Whether or not to generate figures. Default False.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            "--path",
            type=str,
            help="The absolute path to the directory containing the fepout files",
            default=".",
        )
        self.add_argument(
            "--fepoutre",
            type=str,
            help="A regular expression that matches the fepout files to be parsed.",
            default="*.fep*",
        )
        self.add_argument(
            "--replicare",
            type=str,
            help="A regular expression that matches the replica directories",
            default="Replica?",
            required=True,
        )
        self.add_argument(
            "--temperature",
            type=float,
            help="The temperature at which the FEP was run.",
            default=303.15,
        )
        self.add_argument(
            "--detect_equilibrium",
            help="Flag for automated equilibrium detection.",
            default=False,
            action="store_true",
        )
        self.add_argument(
            "--fittingMethod",
            type=str,
            help="Method for fitting the forward-backward discrepancies (hysteresis)."
            + "LS=least squares, ML=maximum likelihood Default: LS",
            default="LS",
        )
        self.add_argument(
            "--max_size",
            type=float,
            help="Maximum total file size in GB."
            + "This is MUCH less than the required RAM. Default: 1",
            default=1,
        )
        self.add_argument(
            "--make_figures",
            help="Flag to generate figures with additional diagnostics.",
            default=False,
            action="store_true",
        )


KILO = 1000


@dataclass(slots=True)
class AFEPArguments:
    """A more readable/testable format for CLI arguments"""

    dataroot: Path
    replica_pattern: str
    filename_pattern: str
    temperature: float
    detect_equilibrium: bool
    make_figures: bool
    RT_kcal_per_mol: float = 0
    replicas: list[str] = None

    @classmethod
    def from_AFEPArgumentParser(cls, parser: AFEPArgumentParser):
        """Unpack arguments from an AFEPArgumentParser object.

        Args:
            parser (AFEPArgumentParser): An AFEPArgumentParser object.

        Returns:
            AFEPArguments: The unpacked arguments.
        """
        args = parser.parse_args()

        dataroot = Path(args.path)
        replica_pattern = args.replicare
        filename_pattern = args.fepoutre

        detect_equilibrium = args.detect_equilibrium

        return cls(
            dataroot,
            replica_pattern,
            filename_pattern,
            args.temperature,
            detect_equilibrium,
            args.make_figures,
        )

    def __post_init__(self) -> None:
        """Get RT, standardize and sort replica names"""
        self.RT_kcal_per_mol = R / (KILO * calorie) * self.temperature
        if self.replica_pattern == '':
            self.replicas = ['.']
        else:
            self.replicas = natsorted(
                [rep.stem for rep in self.dataroot.glob(self.replica_pattern)])
