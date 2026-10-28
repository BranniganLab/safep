from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.testing.compare import compare_images

from safep.plotting import plot_titration


def test_plot_titration(tmp_path):
    concentrations = np.logspace(0, 6, 1000)
    reference = Path(__file__).parent / "test_plotting.test_plot_titration.approved.png"
    actual = "test_plot_titration.png"

    fig, axis = plt.subplots(figsize=(10, 6))
    try:
        plot_titration(axis, concentrations, -4, 0.2, 0.59)
        fig.tight_layout()
        fig.savefig(actual)
    finally:
        plt.close(fig)

    assert compare_images(str(reference), str(actual), tol=0) is None
