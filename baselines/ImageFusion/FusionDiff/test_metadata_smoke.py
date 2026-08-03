import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from metadata_smoke_common import run_smoke
def test_metadata_smoke(): run_smoke()
