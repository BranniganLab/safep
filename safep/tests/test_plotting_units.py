"""Unit checks for plots made from reduced free-energy estimates."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from safep.AFEP_parse import do_per_lambda_convergence_shared_axes
from safep.plotting import plot_general


@pytest.fixture
def estimates():
    cumulative = pd.DataFrame(
        {("BAR", "f"): [0.0, 1.0, 3.0],
         ("BAR", "errors"): [0.0, 0.2, 0.4]},
        index=[0.0, 0.5, 1.0],
    )
    per_window = pd.DataFrame(
        {("BAR", "df"): [1.0, 2.0],
         ("BAR", "ddf"): [0.2, 0.3],
         ("EXP", "dG_f"): [0.4, 0.8],
         ("EXP", "dG_b"): [-0.2, -0.4],
         ("EXP", "difference"): [0.2, 0.4]},
        index=[0.25, 0.75],
    )
    return cumulative, per_window


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("axis_index, errors", [(0, [0.0, 0.2, 0.4]),
                                                (1, [0.2, 0.3])])
def test_general_bar_error_bars_use_same_units_as_values(estimates, axis_index, errors):
    """
    Given estimates and errors in reduced units
    When the general plot is generated
    Then the error bars should be in kcal/mol
    """
    cumulative, per_window = estimates
    RT = 0.6  # kcal/mol per reduced energy unit
    _, axes = plot_general(cumulative, None, per_window, None, RT)

    error_bars = axes[axis_index].containers[0].lines[2][0].get_segments()
    plotted_errors = [(segment[1, 1] - segment[0, 1]) / 2 for segment in error_bars]
    np.testing.assert_allclose(plotted_errors, np.asarray(errors) * RT, atol=1e-12, err_msg="Cumulative plot's error bars don't match expectations.")


def test_general_hysteresis_converts_values_to_kcal_per_mol(estimates):
    """
    Given estimates and errors in reduced units
    When the general plot is generated
    Then the hysteresis data should be in kcal/mol
    """
    cumulative, per_window = estimates
    RT = 0.6
    _, axes = plot_general(cumulative, None, per_window, None, RT, hysttype="lines")
    hyst_ax = axes[2]

    assert "kcal/mol" in hyst_ax.get_ylabel()
    np.testing.assert_allclose(hyst_ax.lines[0].get_ydata(), [0.2 * RT, 0.4 * RT])

class MockArgs():
    def __init__(self, RT):
        self.RT_kcal_per_mol = RT
class MockArgs():
    def __init__(self, RT):
        self.RT_kcal_per_mol = RT

def test_shared_replica_hysteresis_plots_converts_units(estimates):
def test_shared_replica_hysteresis_plots_converts_units(estimates):
    """
    Given estimates and errors in reduced units
    When the shared hyst_ax is generated
    Then the hysteresis data should be in kcal/mol
    """
    _, per_window = estimates
    run = SimpleNamespace(per_window=per_window, color="blue")
    RT = 0.6
    args = MockArgs(RT)
    _, (hyst_ax, _) = do_per_lambda_convergence_shared_axes(args, {"replica": run}, 0.0, 0.0)
    args = MockArgs(RT)
    _, (hyst_ax, _) = do_per_lambda_convergence_shared_axes(args, {"replica": run}, 0.0, 0.0)

    assert "kcal/mol" in hyst_ax.get_ylabel()
    np.testing.assert_allclose(hyst_ax.lines[0].get_ydata(), np.asarray([0.2, 0.4])*RT)
