import pytest
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
    config = {"stepsperstage": 5000000, "stages": 20, "initialWall": 5, "finalWall": 4, "initialequil": 250000,
              "spring": 200}
    return config

def test_read_colvars_traj(pruned_traj):
    assert pruned_traj.loc[19800, "DBC"] == 4.70510005898365e+00

def test_get_stages(pruned_traj, config):
    pruned_traj.get_stages(config)
    assert pruned_traj.loc[105247700, "stage"] == 20
    assert pruned_traj.loc[99800, "stage"] == 0

def test_get_wall_position(pruned_traj, config):
    pruned_traj.get_wall_position(config)
    assert pruned_traj.wall_position.iloc[-1] == 4, "Initial wall position should be 4"
    assert pruned_traj.wall_position.iloc[0] == 5, "Final wall position should be 5"