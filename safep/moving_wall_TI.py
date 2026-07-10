"""Utilities for analyzing moving-wall thermodynamic integration (TI) from NAMD colvars trajectories.

This module provides helpers to:
- Read a NAMD-style moving wall configuration file.
- Load a Colvars trajectory into a pandas DataFrame subclass with convenience methods
  for computing stages, wall positions, and restraint forces.
- Compute per-stage free energy gradients and integrate them to obtain the total
  free energy change using the trapezoidal rule.

The public entrypoint `main` can be used as a CLI to produce a gradients CSV and
print the total free energy change.
"""

import numpy as np

import pandas as pd
from pathlib import Path
from argparse import ArgumentParser


class MovingWallConfig(dict):
    """Contract specification for moving wall infrastructure.

    This is a labile class - it will be replaced when we move to the newer
    moving walls infrastructure in Colvars.

    Caution: only validates inputs on read.
    Switch to UserDict if this gets expanded to other uses.

    Public Methods:
        from_namd_config_file(cls, config_path)
    """
    _required_keys = [
        "stepsperstage",
        "stages",
        "initialequil",
        "initialWall",
        "finalWall",
    ]
    def __init__(self, input_dict: dict):
        super().__init__(input_dict)
        self._validate()

    def _validate(self):
        for key in self._required_keys:
            if key not in self.keys():
                raise ValueError(f"NAMD config missing {key}")

    @classmethod
    def from_namd_config_file(cls, config_path: Path|str) -> dict:
        """Parse a namd config file (or tcl file) to get variable name-value pairs

        Parses all `set VarName VarVal` pairs into a dictionary.

        Arguments:
             config (Path): path to the NAMD config file

        Returns:
            dict: dictionary of all Var: Val pairs
        """
        with open(config_path, encoding="UTF8") as f:
            lines = f.readlines()
        config = {}
        for line in lines:
            if "set" in line:
                try:
                    _, key, value = line.strip().split(" ")
                    config[key] = value
                except ValueError:
                    print(f"bad line: {line}")
                    continue

        for key, value in config.items():
            if value.isnumeric():
                config[key] = float(value)
        return cls(config)

class ColvarsTraj(pd.DataFrame):
    """Container for a Colvars trajectory for moving wall TI

    Public Methods:
        read_colvars_traj(cls, traj_path)
        get_stages(self, config)
        get_wall_position(self, config)
        get_force(self, config)
        see also pandas.DataFrame

    Attributes:
        see pandas.DataFrame
    """
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)

    # Specify attributes to be carried over to new instances
    _metadata = []

    @property
    def _constructor(self):
        """preserve custom class after operations like group"""
        return ColvarsTraj

    @classmethod
    def read_colvars_traj(cls, traj_path: Path) -> pd.DataFrame:
        """Parse a colvars trajectory file

        Arguments:
            traj_path (Path): path to the trajectory file

        Returns:
            ColvarsTraj: the trajectory with step as index and each CV as a column
        """
        with open(traj_path, encoding="UTF8") as f:
            first_line = f.readline()
        header = first_line.strip().split()[1:]
        traj = pd.read_csv(traj_path, sep="\\s+", header=None, engine="python", comment="#")
        traj.columns = header
        traj.set_index("step", inplace=True)
        return cls(traj)

    def get_stages(self, config: MovingWallConfig) -> None:
        """Determine moving wall position using a NAMD config dictionary

        Args:
            config (dict): moving wall parameters including

        Returns:
            None

        Side Effects:
            Adds a stage column to self and populates it with the presumptive stage number
        """
        steps_per_stage = config["stepsperstage"]
        stages = config["stages"]
        initialequil = config["initialequil"]
        self["stage"] = 0

        mask = self.index > initialequil
        steps = self.index[mask]
        self.loc[mask, "stage"] = (steps-initialequil)//steps_per_stage
        if np.any(self["stage"] > stages):
            print("WARNING: Found more steps than should be present given the number of stages")

    def get_wall_position(self, config: MovingWallConfig) -> None:
        if "stage" not in self.columns:
            self.get_stages(config)
        initial_wall = config["initialWall"]
        final_wall = config["finalWall"]
        stages = config["stages"]
        self["wall_position"] = self.stage/stages * (final_wall - initial_wall) + initial_wall

    def get_force(self, config: MovingWallConfig) -> None:
        if "wall_position" not in self.columns:
            self.get_wall_position(config)
        k = config["spring"]
        mask = self.DBC > self.wall_position
        self["force"] = 0.0
        compute_force = lambda sample: k * (sample["wall_position"] - sample["DBC"])
        self.loc[mask, "force"] = self.loc[mask].apply(compute_force, axis=1)


def get_total_free_energy(gradients) -> float:
    total = 0
    for i in range(1, len(gradients)):
        dw = gradients.wall_position.iloc[i] - gradients.wall_position.iloc[i-1]
        total += 0.5 * dw * (gradients.dUdw.iloc[i]+gradients.dUdw.iloc[i-1])
    return total

def get_free_energy_gradients(colvars_traj: ColvarsTraj, config: MovingWallConfig) -> pd.DataFrame:
    if "force" not in colvars_traj.columns:
        colvars_traj.get_force(config)
    all_means = colvars_traj.groupby("stage").mean()
    gradients = all_means.loc[:, ["wall_position", "force"]]
    gradients.rename(columns={"force": "dUdw"}, inplace=True)
    return gradients

def main(config_path, colvars_traj_path, output_prefix):
    config = MovingWallConfig.from_namd_config_file(config_path)
    colvars_traj = ColvarsTraj.read_colvars_traj(colvars_traj_path)
    gradients = get_free_energy_gradients(colvars_traj, config)
    dG = get_total_free_energy(gradients)
    minwall = min(config["initialWall"], config["finalWall"])
    maxwall = max(config["initialWall"], config["finalWall"])
    print(f"The total free energy change going from {minwall} to {maxwall} is {dG} kcal/mol")
    gradients.to_csv(output_prefix+"_gradients.csv", index=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_path", type=Path)
    parser.add_argument("colvars_traj_path", type=Path)
    parser.add_argument("output_prefix", type=str, default="./log")
    args = parser.parse_args()

    main(args.config_path, args.colvars_traj_path, args.output_prefix)

