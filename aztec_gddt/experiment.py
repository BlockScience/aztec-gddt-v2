import pandas as pd
from aztec_gddt.structure import MODEL_BLOCKS
from aztec_gddt.utils import sim_run

def test_run() -> pd.DataFrame:
    """Function which runs the cadCAD simulations

    Returns:
        DataFrame: A dataframe of simulation data
    """
    from aztec_gddt.default_params import DEFAULT_INITIAL_STATE, DEFAULT_PARAMS
    # The number of timesteps for each simulation to run
    N_TIMESTEPS = 3000

    # The number of monte carlo runs per set of parameters tested
    N_samples = 1
    # %%
    # Get the sweep params in the form of single length arrays
    sweep_params = {k: [v] for k, v in DEFAULT_PARAMS.items()}

    # Load simulation arguments
    sim_args = (DEFAULT_INITIAL_STATE, sweep_params, MODEL_BLOCKS, N_TIMESTEPS, N_samples)

    # Run simulation
    sim_df = sim_run(*sim_args)
    return sim_df

