from pathlib import Path
import pytest

from safep.fileIO import get_num_regex

@pytest.mark.parametrize(["regex", "expected_value"],
                         [(r"(upperWalls)\s+(\d+)", 6),
                          (r"(forceConstant)\s+(\d+(?:\.\d+)?)", 200.0)]
                          )
def test_get_num_regex_extracts_the_right_number(regex, expected_value):
    colvars_file = Path(__file__).parent / "data" / "toy.colvars"

    value = get_num_regex(regex, colvars_file)

    assert float(value) == expected_value

