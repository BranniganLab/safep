import numpy as np
from approvaltests import verify
from safep.AFEP_parse import  COLORS, get_summary_statistics, AFEPArguments, get_sterr
from safep.fepruns import process_replicas
import pytest
from pathlib import Path

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

def test_sterr_of_five_numbers_is_correct():
    dGs = [1,2,3,4,5]
    errors = [1,1,1,1,1]
    sterr = get_sterr(dGs, errors)
    assert not np.isclose(sterr, 1.58113883), "Got standard deviation, not standard error"
    assert np.isclose(sterr, 0.7071067812), "Wrong standard error"

def test_sterr_of_two_numbers_propagates_error():
    dGs = [3,4]
    errors = [1,2]
    sterr = get_sterr(dGs, errors)
    assert not np.isclose(sterr, 0.7071067812), "Got standard deviation, not propagated error"
    assert not np.isclose(sterr, 0.5), "Got standard error. Standard error of two numbers is a math crime. The authorities have been informed."
    assert np.isclose(sterr, 2.236067977), "Error not propagated correctly."
