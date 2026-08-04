from approvaltests import verify
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
from approvaltests import verify
from approvaltests.namer import NamerFactory
from safep.AFEP_parse import  COLORS, get_summary_statistics, AFEPArguments, get_sterr
from safep.fepruns import process_replicas, FepRun
import pytest
from pathlib import Path
import shutil

@pytest.fixture
def itcolors():
    return iter(COLORS)

@pytest.fixture
def test_data_path():
    return Path(__file__).parent/"../../Sample_Notebooks/Sample_Data"

@pytest.fixture(scope="function", params=["idws", "forward"])
def afep_args(test_data_path, tmp_path, request):
    test_directory = Path(tmp_path)/"test"
    replica1 = test_directory/"Replica1"

    shutil.copytree(test_data_path, replica1)
    cached_file = replica1 / "decorrelated.csv"
    if cached_file.exists():
        Path.unlink(cached_file)

    replica2 = test_directory/"Replica2"
    replica2.symlink_to(replica1)
    replica3 = test_directory/"Replica3"
    replica3.symlink_to(replica1)

    prefix = request.param
    return AFEPArguments(dataroot = test_directory,
                        replica_pattern = "Replica*",
                        replicas = None,
                        filename_pattern = f"{prefix}*.fep*",
                        temperature = 303.15,
                        detect_equilibrium = True,
                        make_figures = False)

@pytest.fixture
def fepruns(afep_args, itcolors) -> dict[str, FepRun]:
    return process_replicas(afep_args, itcolors)


def test_summary(afep_args, fepruns, request):
    summary, dGs, mean, sterr = get_summary_statistics(afep_args, fepruns)
    test_id = request.node.callspec.id
    verify(summary, options=NamerFactory.with_parameters(test_id))

def test_u_nk(fepruns, request):
    """
    This is both a test and an example for testing numerical data with numpy allclose while
    remaining consistent with the approvaltests paradigm of having an "approved" vs "received" file.

    Given: A set of energy differences, u_nk
    When: compared to an approved array
    Expect: the two arrays to be within machine tolerance (numpy allclose)
    """
    test_id = request.node.callspec.id
    received = fepruns["Replica1"].u_nk
    ref_path = Path(__file__).parent / f"test_afep_parse.test_u_nk.{test_id}.approved.txt"
    expected_u_nk = pd.read_csv(ref_path)
    expected_u_nk.columns = expected_u_nk.columns.astype(float)

    pd.testing.assert_frame_equal(
        received.reset_index(drop=True), # Ensure index doesn't block comparison
        expected_u_nk,
        atol=1e-6,
        check_column_type=False,
    )
    try:
        max_error = ((expected_u_nk - received).abs()).max().max()
    except ValueError:
        max_error = "incalculable"
    print(f"U_nk does not match approved. Max error: {max_error}. "
         f"To approve the current version, rename "
         f"test_afep_parse.test_u_nk.received.txt to test_afep_parse.test_u_nk.approved.txt "
         f"and commit the result")

def test_sterr_of_five_numbers_is_correct():
    dGs = [1,2,3,4,5]
    errors = [1,1,1,1,1]
    sterr = get_sterr(dGs, errors)
    assert not np.isclose(sterr, 1.58113883), "Got standard deviation, not standard error"
    assert np.isclose(sterr, 0.7071067812), f"Got: {sterr}. Expected: 0.7071067812"

def test_sterr_of_two_numbers_propagates_error():
    dGs = [3,4]
    errors = [1,2]
    sterr = get_sterr(dGs, errors)
    assert not np.isclose(sterr, 0.7071067812), "Got standard deviation, not propagated error"
    assert not np.isclose(sterr, 0.5), "Got standard error. Standard error of two numbers is a math crime. The authorities have been informed."
    assert np.isclose(sterr, 2.236067977), "Error not propagated correctly."

def test_cached_fepruns_match_expectations(fepruns: dict[str, FepRun], tmp_path: Path):
    """
    Given a set of fepruns
    When written to file and read back
    Expect the feprun as-read to be identical to the feprun as-written
    """
    for key, fr in fepruns.items():
        fr.to_dir(tmp_path/key)
        test_fr= FepRun.from_dir(tmp_path/key)
        for name in ["u_nk", "per_window", "cumulative", "forward", "forward_error", "backward", "backward_error", "per_lambda_convergence", "color"]:
            test = getattr(test_fr, name)
            canonical = getattr(fr, name)
            if isinstance(test, pd.DataFrame):
                assert_frame_equal(test, canonical)
            elif isinstance(test, np.ndarray):
                assert np.allclose(test, canonical)
            else:
                assert test == canonical

