import numpy as np
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
    """
    This is both a test and an example for testing numerical data with numpy allclose while
    remaining consistent with the approvaltests paradigm of having an "approved" vs "received" file.

    Given: A set of energy differences, u_nk
    When: compared to an approved array
    Expect: the two arrays to be within machine tolerance (numpy allclose)
    """
    u_nk = fepruns["Replica1"].u_nk
    received = np.asarray(u_nk)
    received = np.concatenate([[list(u_nk.columns)], received], 0)
    np.savetxt(Path(__file__).parent/"test_afep_parse.test_u_nk.received.txt", received, delimiter=",")
    approved = np.genfromtxt(Path(__file__).parent/"test_afep_parse.test_u_nk.approved.txt", delimiter=",")
    assert np.allclose(approved, received, equal_nan=True), (
         f"U_nk does not match approved. Max error: {np.max(np.abs(approved - received))}. "
         f"To approve the current version, rename "
         f"test_afep_parse.test_u_nk.received.txt to test_afep_parse.test_u_nk.approved.txt "
         f"and commit the result")