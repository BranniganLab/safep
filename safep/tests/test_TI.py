import pytest
from safep.moving_wall_TI import read_namd_conf_moving_wall, ColvarsTraj
from pathlib import Path

def test_read_namd_conf():
    config = read_namd_conf_moving_wall(Path(__file__).parent/"data/job_1.namd")
    assert config['stepsperstage'] == 5000000

@pytest.fixture(scope="module")
def pruned_traj() -> ColvarsTraj:
    return ColvarsTraj.read_colvars_traj(Path(__file__).parent/"data/pruned.colvars.traj")

def test_read_colvars_traj(pruned_traj):
    assert pruned_traj.loc[19800, "DBC"] == 4.70510005898365e+00

def test_get_stages(pruned_traj):
    config = {"stepsperstage": 1000, "stages": 20}
    pruned_traj.get_stages(config)
    assert pruned_traj.loc[19800, "stage"] == 19
