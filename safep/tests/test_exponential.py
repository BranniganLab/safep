import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from safep import get_exponential
from alchemlyb.parsing import namd

def test_exponential_averaging():
    """
    Given a dummy dataset of all 1s
    When exponential averaged
    Expect all 1s
    """
    u_nk = pd.read_csv(Path(__file__).parent/"test_u_nk.csv", index_col=(0,1))
    u_nk.columns = pd.to_numeric(u_nk.columns)
    l, l_mid, dG_f, dG_b = get_exponential(u_nk)

    assert np.allclose(dG_f, 1), f"dG_f:{dG_f[0]}, dG_b:{dG_b[0]}"
    assert np.allclose(dG_b, 1), f"dG_f:{dG_f[0]}, dG_b:{dG_b[0]}"

def test_exponential_averaging_diff_fwd_bwd():
    """
    Given a dummy dataset of all 1s forward and -2s backward
    When exponential averaged
    Expect all 1s forward and -2s backward
    """
    u_nk = pd.read_csv(Path(__file__).parent/"test_diff_fwd_bwd.csv", index_col=(0,1))
    u_nk.columns = pd.to_numeric(u_nk.columns)
    l, l_mid, dG_f, dG_b = get_exponential(u_nk)

    assert np.allclose(dG_f, 1), f"dG_f:{dG_f[0]}, dG_b:{dG_b[0]}"
    assert np.allclose(dG_b, -2), f"dG_f:{dG_f[0]}, dG_b:{dG_b[0]}"

def test_artificially_bad_window():
    """
    Given a dataset with an artificial bad window
    When exponential averaged
    Expect the forward spike to come after the backward spike
    """
    datapath = Path(__file__).parent/"../../Sample_Notebooks/Replica1"
    u_nk = namd.extract_u_nk(list(datapath.glob("idws*.fepout")), 303.15)
    u_nk = u_nk.sort_index(axis=0, level=1).sort_index(axis=1)
    u_nk.loc[(slice(None), 0.3), slice(None)] *= 10
    l, l_mid, dG_f, dG_b = get_exponential(u_nk)

    outlier_forward = np.argmax(np.abs(dG_f))
    outlier_backward = np.argmax(np.abs(dG_b))
    assert outlier_forward == outlier_backward + 1, "Worst window misidentified"
