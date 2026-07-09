from approvaltests import verify
import pytest
from matplotlib.testing.compare import compare_images
from safep.RFEP_analysis import main
from pathlib import Path
from safep.moving_wall_TI import main as moving_wall

def test_RFEP_main_out(capsys):
    main(Path(__file__).parent/"RFEP_decouple.log")
    captured = capsys.readouterr()
    verify(captured.out)

def test_RFEP_main_err(capsys):
    main(Path(__file__).parent / "RFEP_decouple.log")
    captured = capsys.readouterr()
    verify(captured.err)

def test_RFEP_figure():
    main(Path(__file__).parent / "RFEP_decouple.log")
    ref = Path(__file__).parent / "test_rfep_parse.test_RFEP_figure.approved.png"
    actual = Path("RFEP_decouple_figures.png")
    compare_images(ref, actual, tol=1e-5)
    actual.unlink()

def test_moving_wall(capsys):
    config = Path(__file__).parent / "data" / "job_1.namd"
    colvars_traj = Path(__file__).parent / "data" / "pruned.colvars.traj"
    output_prefix = "tmp.moving_wall"
    moving_wall(config, colvars_traj, output_prefix)
    captured = capsys.readouterr()
    lines = captured.out.split('\n')
    assert "0.5954" in lines[-2], "Expected total free energy to be about 0.5954"
    assert "6.0 to 8.0" in lines[-2], "Expected the wall to move from 6 to 8"