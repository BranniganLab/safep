from approvaltests.approvals import verify
from safep.fileIO import parse_Colvars_log
from pathlib import Path

SAMPLE_RFEP_LOG = Path("../../Sample_Notebooks/RFEP_decouple.log")

def test_parsing():
    result = parse_Colvars_log(SAMPLE_RFEP_LOG)
    verify(result)
