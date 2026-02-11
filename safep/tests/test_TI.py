import pytest
import numpy as np
from safep.moving_wall_TI import read_namd_conf_moving_wall, ColvarsTraj
from pathlib import Path

def test_read_namd_conf():
    config = read_namd_conf_moving_wall(Path(__file__).parent/"data/job_1.namd")
    assert config['stepsperstage'] == 5000000

@pytest.fixture(scope="module")
def pruned_traj() -> ColvarsTraj:
    return ColvarsTraj.read_colvars_traj(Path(__file__).parent/"data/pruned.colvars.traj")

@pytest.fixture(scope="module")
def config() -> dict:
    config = read_namd_conf_moving_wall(Path(__file__).parent / "data/job_1.namd")
    return config

def test_read_colvars_traj(pruned_traj):
    assert pruned_traj.loc[19800, "DBC"] == 4.70510005898365e+00

def test_get_stages(pruned_traj, config):
    pruned_traj.get_stages(config)
    assert pruned_traj.loc[105247700, "stage"] == 20
    assert pruned_traj.loc[99800, "stage"] == 0

def test_get_wall_position(pruned_traj, config):
    pruned_traj.get_wall_position(config)
    assert pruned_traj.wall_position.iloc[0] == 8, "Initial wall position should be 8"
    assert pruned_traj.wall_position.iloc[-1] == 6, "Final wall position should be 6"

def test_get_gradient(pruned_traj):
    pruned_traj.get_gradient(config)
    assert pruned_traj.gradient.iloc[0] == 0, "Initial gradient should be 0, because initial DBC is less than the wall"
    assert np.isclose(pruned_traj.gradient.loc[1229700], -4.19983056656), "Gradient when DBC=8.041... should be -4.19..."