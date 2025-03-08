from approvaltests.approvals import verify
from safep.fileIO import parse_Colvars_log
from pathlib import Path
import pytest
from collections import namedtuple
import json

SAMPLE_RFEP_LOG = Path("../../Sample_Notebooks/RFEP_decouple.log")

@pytest.fixture
def parse_result():
    ParseResult = namedtuple('ParseReult', 'global_conf colvars biases TI_traj')
    return ParseResult(*parse_Colvars_log(SAMPLE_RFEP_LOG))

def test_global_config(parse_result):
    test_value = json.loads(json.dumps(parse_result.global_conf, sort_keys=True))
    verify(test_value)

def test_colvars(parse_result):
    test_value = json.loads(json.dumps(parse_result.colvars, sort_keys=True))
    verify(test_value)

def test_biases(parse_result):
    test_value = json.loads(json.dumps(parse_result.biases, sort_keys=True))
    verify(test_value)

def test_TI_traj(parse_result):
    test_value = json.loads(json.dumps(parse_result.TI_traj, sort_keys=True))
    verify(test_value)

