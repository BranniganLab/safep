from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.testing.compare import compare_images

from safep.plotting import SampleAxes, plot_samples


@pytest.fixture
def u_nk():
    index = pd.MultiIndex.from_tuples(
        [(0, 0.0), (1, 0.0), (2, 0.0), (0, 0.5), (1, 0.5)],
        names=["time", "fep-lambda"],
    )
    return pd.DataFrame({0.0: [1, 2, 3, 4, 5]}, index=index)


@pytest.mark.parametrize("entry_point", ["method", "function"])
def test_plot_samples_approval(u_nk, entry_point, tmp_path):
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    fig.subplots_adjust(left=0.2, bottom=0.25)
    try:
        if entry_point == "method":
            sample_ax = SampleAxes(fig, ax.get_position())
            fig.delaxes(ax)
            fig.add_axes(sample_ax)
            sample_ax.plot_samples(u_nk, color="red", label="Samples")
        else:
            sample_ax = plot_samples(ax, u_nk, color="red", label="Samples")
            assert sample_ax is ax

        if entry_point == "method":
            assert isinstance(sample_ax, SampleAxes)
        actual = tmp_path / "test_plotting.test_plot_samples.received.png"
        fig.savefig(actual)
        approved = Path(__file__).with_name("test_plotting.test_plot_samples.approved.png")
        assert compare_images(str(approved), str(actual), tol=0) is None
    finally:
        plt.close(fig)
