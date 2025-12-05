from approvaltests import verify
from safep.AFEP_parse import  COLORS, get_summary_statistics, AFEPArguments
from safep.fepruns import process_replicas
import pytest
from pathlib import Path

@pytest.fixture
def itcolors():
    return iter(COLORS)

@pytest.fixture
def afep_args():
    return AFEPArguments(dataroot = Path("../../Sample_Notebooks"),
                        replica_pattern = "Replica*",
                        replicas = None,
                        filename_pattern = "*.fep*",
                        temperature = 303.15,
                        detectEQ = True,
                        makeFigures = False)

@pytest.fixture
def fepruns(afep_args, itcolors):
    return process_replicas(afep_args, itcolors)

def test_summary(afep_args, fepruns):
    summary, dGs, mean, sterr = get_summary_statistics(afep_args, fepruns)
    verify(summary)

def test_u_nk(fepruns):
    u_nk = fepruns["Replica1"].u_nk
    string = u_nk.to_csv(index=False)
    verify(string)