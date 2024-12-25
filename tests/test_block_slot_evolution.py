import pandas as pd
from aztec_gddt.experiment import test_run
from aztec_gddt.types import Agent
import pytest as pt
import pandera as pa
import pytest

@pt.fixture(scope="module")
def sim_df() -> pd.DataFrame:
    return test_run()
