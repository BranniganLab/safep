from safep.moving_wall_TI import read_namd_conf_moving_wall, read_colvars_traj
from pathlib import Path

def test_read_namd_conf():
    config = read_namd_conf_moving_wall(Path(__file__).parent/"data/job_1.namd")
    assert config['stepsperstage'] == 5000000

def test_read_colvars_traj():
    traj = read_colvars_traj(Path(__file__).parent/"data/pruned.colvars.traj")
    assert traj.loc[19800, "DBC"] == 4.70510005898365e+00