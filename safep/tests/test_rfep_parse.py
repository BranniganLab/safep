from approvaltests import verify
import pytest
from safep.RFEP_analysis import main

pytest.mark.xfail
def test_RFEP_main_out():
    captured = capsys.readouterr()
    verify captured.out

def test_RFEP_main_err():
    captured = capsys.readouterr()
    verify captured.err