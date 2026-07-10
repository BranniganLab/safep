import pytest
import numpy as np
from safep.moving_wall_TI import MovingWallConfig, ColvarsTraj, get_free_energy_gradients, \
    get_total_free_energy, main as moving_wall
from pathlib import Path

def test_read_namd_conf():
    """
    Given a config file with the line `set stepsperstage 5000000`
    When parsed
    Expect a dictionary with the key `stepsperstage` with value 5000000
    """
    config = MovingWallConfig.from_namd_config_file(Path(__file__).parent / "data/job_1.namd")
    assert config['stepsperstage'] == 5000000

@pytest.fixture(scope="module")
def pruned_traj() -> ColvarsTraj:
    """Parsed colvars traj"""
    return ColvarsTraj.read_colvars_traj(Path(__file__).parent/"data/pruned.colvars.traj")

@pytest.fixture(scope="module")
def config() -> dict:
    """Parsed namd config"""
    config = MovingWallConfig.from_namd_config_file(Path(__file__).parent / "data/job_1.namd")
    return config

def test_read_colvars_traj(pruned_traj):
    """Spot-check trajectory parsing"""
    assert pruned_traj.loc[19800, "DBC"] == 4.70510005898365e+00

def test_get_stages(pruned_traj, config):
    """Spot-check stage assignment"""
    pruned_traj.get_stages(config)
    assert pruned_traj.loc[105247700, "stage"] == 20
    assert pruned_traj.loc[99800, "stage"] == 0

def test_get_wall_position(pruned_traj, config):
    """Spot-check wall positions"""
    pruned_traj.get_wall_position(config)
    assert pruned_traj.wall_position.iloc[0] == 8, "Initial wall position should be 8"
    assert pruned_traj.wall_position.iloc[-1] == 6, "Final wall position should be 6"

def test_get_force(pruned_traj, config):
    """Spot-check force assignment"""
    pruned_traj.get_force(config)
    assert pruned_traj.force.iloc[0] == 0, "Initial Force should be 0, because initial DBC is less than the wall"
    assert np.isclose(pruned_traj.force.loc[1229700], -8.39966113312), "Force when DBC=8.041... should be -8.4..."
    assert np.isclose(pruned_traj.force.loc[104797700], -3.83625847005), "Force when DBC=6.02... and wall=6, should be -3.8..."


def test_get_free_energy_gradients(pruned_traj, config):
    """
    Given the trajectory and config
    When gradients are computed
    Expect all negative gradients
    Expect the correct number of stages
    """
    gradients = get_free_energy_gradients(pruned_traj, config)
    # Gradients are always in the expanding direction, so they should all be negative
    assert np.all(gradients.dUdw.values <= 0), "Positive gradients found"
    assert len(gradients) == config["stages"]+1, "Too many stages with gradients"

def test_toy_data():
    """
    Given some toy data with known sum of gradients
    When get_free_energy_gradients is called
    Expect the sum of gradients to be correct
    """
    traj = ColvarsTraj.read_colvars_traj(Path(__file__).parent/"data/toy.colvars.traj")
    config = MovingWallConfig.from_namd_config_file(Path(__file__).parent / "data/toy.namd")
    gradients = get_free_energy_gradients(traj, config)
    assert gradients.dUdw.sum() == -60, "One or more toy data gradients are wrong"

def test_toy_dG_release():
    """
    Given some toy data with known dG_DBC
    When get_total_free_energy is called
    Expect the total free energy to be correct
    """
    traj = ColvarsTraj.read_colvars_traj(Path(__file__).parent/"data/toy.colvars.traj")
    config = MovingWallConfig.from_namd_config_file(Path(__file__).parent / "data/toy.namd")
    gradients = get_free_energy_gradients(traj, config)
    dG = get_total_free_energy(gradients)
    assert dG == -30, "Toy data is returning the wrong free energy."


def test_moving_wall(capsys):
    """Approval test for CLI main"""
    config = Path(__file__).parent / "data" / "job_1.namd"
    colvars_traj = Path(__file__).parent / "data" / "pruned.colvars.traj"
    output_prefix = "tmp.moving_wall"
    moving_wall(config, colvars_traj, output_prefix)
    captured = capsys.readouterr()
    lines = captured.out.split('\n')
    assert "0.5954" in lines[-2], "Expected total free energy to be about 0.5954"
    assert "6.0 to 8.0" in lines[-2], "Expected the wall to move from 6 to 8"
