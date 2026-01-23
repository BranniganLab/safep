from approvaltests import verify
import pytest
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