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


RT = 0.59
@pytest.mark.parametrize(["dG", "expected_kd"],
                         [(0, 1.e6),
                          (-1e10, 0),
                          (RT, np.e*1.e6)])
def test_dissociation_constant_obeys_boltzmann(dG, expected_kd):
    kd = get_dissociation_constant(dG, RT)
    assert np.isclose(kd, expected_kd)
