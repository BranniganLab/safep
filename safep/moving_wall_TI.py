import numpy as np

import pandas as pd
from pathlib import Path

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

def read_colvars_traj(traj_path: Path) -> pd.DataFrame:
    with open(traj_path, encoding="UTF8") as f:
        first_line = f.readline()
    header = first_line.strip().split()[1:]
    traj = pd.read_csv(traj_path, sep="\s+", header=None, engine="python", comment="#")
    traj.columns = header
    traj.set_index("step", inplace=True)
    return traj
