
import pandas as pd
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
    u_nk = fepruns["Replica1"].u_nk
    ref_path = Path(__file__).parent / "test_afep_parse.test_u_nk.approved.txt"
    # Fro updating reference data
    # if not ref_path.exists():
    #     u_nk.to_csv(ref_path, index=False)
    #     pytest.fail(f"Reference file created at {ref_path}. Inspect it and re-run test.")

    expected_u_nk = pd.read_csv(ref_path)
    expected_u_nk.columns = expected_u_nk.columns.astype(float)

    pd.testing.assert_frame_equal(
        u_nk.reset_index(drop=True), # Ensure index doesn't block comparison
        expected_u_nk,
        atol=1e-6,
        check_column_type=False
    )
