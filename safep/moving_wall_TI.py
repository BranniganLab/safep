import numpy as np

import pandas as pd
from pathlib import Path
from argparse import ArgumentParser

def read_namd_conf_moving_wall(config: Path) -> dict:
    with open(config, encoding="UTF8") as f:
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
        try:
            config[key] = float(value)
        except ValueError:
            continue
    return config

class ColvarsTraj(pd.DataFrame):
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
        with open(traj_path, encoding="UTF8") as f:
            first_line = f.readline()
        header = first_line.strip().split()[1:]
        traj = pd.read_csv(traj_path, sep="\\s+", header=None, engine="python", comment="#")
        traj.columns = header
        traj.set_index("step", inplace=True)
        return cls(traj)

    def get_stages(self, config: dict) -> None:
        steps_per_stage = config["stepsperstage"]
        stages = config["stages"]
        initialequil = config["initialequil"]
        self["stage"] = 0

        mask = self.index > initialequil
        steps = self.index[mask]
        self.loc[mask, "stage"] = (steps-initialequil)//steps_per_stage
        if np.any(self["stage"] > stages):
            print("WARNING: Found more steps than should be present given the number of stages")

    def get_wall_position(self, config: dict) -> None:
        if "stage" not in self.columns:
            self.get_stages(config)
        initial_wall = config["initialWall"]
        final_wall = config["finalWall"]
        stages = config["stages"]
        self["wall_position"] = self.stage/stages * (final_wall - initial_wall) + initial_wall

    def get_force(self, config: dict) -> None:
        if "wall_position" not in self.columns:
            self.get_wall_position(config)
        k = config["spring"]
        mask = self.DBC > self.wall_position
        self["force"] = 0.0
        compute_force = lambda sample: k * (sample["wall_position"] - sample["DBC"])
        self.loc[mask, "force"] = self.loc[mask].apply(compute_force, axis=1)

    def get_free_energy_gradients(self, config: dict) -> pd.Series:
        if "force" not in self.columns:
            self.get_force(config)
        gradients = self.groupby("stage")["force"].mean()
        return gradients

def main(config_path, colvars_traj_path):
    config = read_namd_conf_moving_wall(config_path)
    colvars_traj = ColvarsTraj.read_colvars_traj(colvars_traj_path)
    gradients = colvars_traj.get_free_energy_gradients(config)
    dG = gradients.sum()
    minwall = min(config["initialWall"], config["finalWall"])
    maxwall = max(config["initialWall"], config["finalWall"])
    print(f"The total free energy change going from {minwall} to {maxwall} is {dG} kcal/mol")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_path", type=Path)
    parser.add_argument("colvars_traj_path", type=Path)
    args = parser.parse_args()

    main(args.config_path, args.colvars_traj_path)

