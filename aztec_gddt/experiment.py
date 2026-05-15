"""
Experiment execution module for Aztec GDDT simulations.

This module provides functions to run cadCAD simulations of the Aztec gddt v2 with 
various parameter configurations. It includes both a simple test run function with default
parameters and a more flexible custom run function that allows parameter sweeping.

"""

import pandas as pd
from pandas import DataFrame

from typing import Dict, List, Optional
from pathlib import Path

from aztec_gddt.structure import MODEL_BLOCKS
from aztec_gddt.utils import sim_run
from aztec_gddt.default_params import *
from aztec_gddt.types import *
from aztec_gddt import DEFAULT_LOGGER

from cadCAD.tools.preparation import sweep_cartesian_product  # type: ignore


def test_run(N_TIMESTEPS=3000) -> pd.DataFrame:
    """
    Run a basic simulation using default parameters for quick testing.

    This function provides a simple way to execute a cadCAD simulation of the Aztec system
    using the default model configuration. It uses a single sample (no Monte Carlo variation)
    and the default model parameters.

    Args:
        N_TIMESTEPS (int, optional): Number of timesteps to run the simulation. 
            Default is 3000, which represents approximately 3000 L1 blocks.

    Returns:
        DataFrame: A dataframe containing the simulation results for each timestep,
            with system state variables tracked over time.
    """
    from aztec_gddt.default_params import DEFAULT_INITIAL_STATE, DEFAULT_PARAMS
    # The number of timesteps for each simulation to run

    # The number of monte carlo runs per set of parameters tested
    # Only 1 sample is used for test runs (no stochastic variation between runs)
    N_samples = 1
    # %%
    # Get the sweep params in the form of single length arrays
    sweep_params = {k: [v] for k, v in DEFAULT_PARAMS.items()}

    # Load simulation arguments
    sim_args = (DEFAULT_INITIAL_STATE, sweep_params, MODEL_BLOCKS, N_TIMESTEPS, N_samples)

    # Execute the simulation and return the resulting DataFrame
    sim_df = sim_run(*sim_args)
    return sim_df


def custom_run(
    initial_state: Optional[ModelState] = None,
    default_params: Optional[ModelParams] = None,
    params_to_modify: Optional[dict[str, List]] = None,
    model_blocks: Optional[list[dict]] = None,
    N_timesteps: int = 100,
    N_samples: int = 1,
) -> DataFrame:
    """
    Run a custom simulation with customizable parameters and model configuration.

    This function provides a flexible way to execute simulations with modified parameters, initial state, or even model structure. It supports parameter sweeping by providing lists of parameter values to test across multiple simulation runs.

    Args:
        initial_state (ModelState, optional): The initial system state. If None, uses 
            DEFAULT_INITIAL_STATE from default_params.py.

        default_params (ModelParams, optional): Base model parameters to use. If None, 
            uses DEFAULT_PARAMS from default_params.py.

        params_to_modify (dict[str, List], optional): Parameters to modify from the defaults.
            For each key in this dict, the corresponding value in default_params will be 
            replaced with the provided value(s). If a list is provided for a parameter,
            cadCAD will perform a parameter sweep across those values.

        model_blocks (list[dict], optional): The model update blocks defining the simulation
            structure. If None, uses MODEL_BLOCKS from structure.py.

        N_timesteps (int, optional): Number of timesteps to run the simulation. Default is 100.

        N_samples (int, optional): Number of Monte Carlo samples for each parameter 
            configuration. Default is 1 (no Monte Carlo variation).

    Returns:
        DataFrame: A dataframe containing the simulation results for each timestep and 
            parameter configuration, with system state variables tracked over time.
    """
    # Set default values if not provided
    if initial_state is None:
        initial_state = DEFAULT_INITIAL_STATE
    if default_params is None:
        default_params = DEFAULT_PARAMS
    if model_blocks is None:
        model_blocks = MODEL_BLOCKS

    # Convert the default parameters to the cadCAD format (lists for parameter sweeping)
    sweep_params = {k: [v] for k, v in default_params.items()}

    # Override default parameter values with those provided in params_to_modify
    if params_to_modify is not None:
        # For each parameter to modify, update the sweep_params dictionary
        for k, v in params_to_modify.items():
            sweep_params[k] = v
            # The following commented code shows an alternative approach that
            # ensures values are always lists (even when a single value is provided)
            # if isinstance(v, list):
            #     sweep_params[k] = v
            # else:
            #     sweep_params[k] = [v]
    else:
        pass

    # Package all simulation arguments for the sim_run function
    sim_args = (initial_state, sweep_params,
                model_blocks, N_timesteps, N_samples)

    # Execute the simulation and return the resulting DataFrame
    sim_df = sim_run(*sim_args)
    return sim_df
