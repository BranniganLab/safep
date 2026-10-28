import numpy as np
import pytest

from safep.estimators import binding_probability, get_dissociation_constant


@pytest.mark.parametrize(["kd", "concentrations", "expectation"],
                        [(0.01, 0.01, 0.5),
                         (0.01, [0,0.01,100000], [0,0.5,1])]
                         )
def test_binding_probability_obeys_hill_equation(kd, concentrations, expectation):
    p = binding_probability(kd, np.asarray(concentrations))
    assert np.allclose(p, expectation)

