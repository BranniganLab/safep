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

def test_get_force(pruned_traj, config):
    pruned_traj.get_force(config)
    assert pruned_traj.force.iloc[0] == 0, "Initial Force should be 0, because initial DBC is less than the wall"
    assert np.isclose(pruned_traj.force.loc[1229700], -8.39966113312), "Force when DBC=8.041... should be -8.4..."
    assert np.isclose(pruned_traj.force.loc[104797700], -3.83625847005), "Force when DBC=6.02... and wall=6, should be -3.8..."


def test_get_free_energy_gradients(pruned_traj, config):
    gradients = pruned_traj.get_free_energy_gradients(config)
    # Gradients are always in the expanding direction, so they should all be negative
    assert np.all(gradients.dUdw.values <= 0), "Positive gradients found"
    assert len(gradients) == config["stages"]+1, "Too many stages with gradients"

def test_toy_data():
    traj = ColvarsTraj.read_colvars_traj(Path(__file__).parent/"data/toy.colvars.traj")
    config = read_namd_conf_moving_wall(Path(__file__).parent/"data/toy.namd")
    gradients = traj.get_free_energy_gradients(config)
    assert gradients.dUdw.sum() == -60, "One or more toy data gradients are wrong"

def test_toy_dG_release():
    traj = ColvarsTraj.read_colvars_traj(Path(__file__).parent/"data/toy.colvars.traj")
    config = read_namd_conf_moving_wall(Path(__file__).parent/"data/toy.namd")
    dG = traj.get_total_free_energy(config)
    assert dG == -30, "Toy data is returning the wrong free energy."