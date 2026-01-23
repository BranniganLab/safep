from approvaltests import verify
import pytest
from matplotlib.testing.compare import compare_images
from safep.RFEP_analysis import main
from pathlib import Path

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
    actual = Path(__file__).parent / "RFEP_decouple_figures.png"
    compare_images(ref, actual, tol=1e-5)
    actual.unlink()