import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from safep import get_exponential


def test_exponential_averaging():
    u_nk = pd.read_csv(Path(__file__).parent/"test_u_nk.csv", index_col=(0,1))
    u_nk.columns = pd.to_numeric(u_nk.columns)
    l, l_mid, dG_f, dG_b = get_exponential(u_nk)

    assert np.allclose(dG_f, 1), f"dG_f:{dG_f[0]}, dG_b:{dG_b[0]}"
    assert np.allclose(dG_b, 1), f"dG_f:{dG_f[0]}, dG_b:{dG_b[0]}"

def test_exponential_averaging_diff_fwd_bwd():
    u_nk = pd.read_csv(Path(__file__).parent/"test_diff_fwd_bwd.csv", index_col=(0,1))
    u_nk.columns = pd.to_numeric(u_nk.columns)
    l, l_mid, dG_f, dG_b = get_exponential(u_nk)

    assert np.allclose(dG_f, 1), f"dG_f:{dG_f[0]}, dG_b:{dG_b[0]}"
    assert np.allclose(dG_b, -2), f"dG_f:{dG_f[0]}, dG_b:{dG_b[0]}"