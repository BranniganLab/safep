from pathlib import Path
import pytest
from approvaltests import verify
import pandas as pd
from safep.AFEP_parse import  COLORS, get_summary_statistics, AFEPArguments
from safep.fepruns import process_replicas
from safep.estimators import do_estimation


@pytest.fixture
def itcolors():
    return iter(COLORS)

@pytest.fixture
def afep_args():
    return AFEPArguments(dataroot = Path(__file__).parent/"../../Sample_Notebooks",
                        replica_pattern = "Replica*",
                        replicas = None,
                        filename_pattern = "*.fep*",
                        temperature = 303.15,
                        detect_equilibrium = True,
                        make_figures = False)

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

def test_estimators(fepruns):
    u_nk = fepruns["Replica1"].u_nk
    per_window, cumulative = do_estimation(u_nk, "both")
    string = "per_window:\n"
    string += per_window.to_csv(index=False)
    string += "\n\ncumulative:\n"
    string += cumulative.to_csv(index=False)
    verify(string)
