"""
Utility functions and helpers for the Aztec GDDT simulation.

This module provides utility functions for:
1. Running and configuring cadCAD simulations
2. Suppressing unnecessary output and progress bars
3. Aggregating policy functions
4. Simplifying the execution of simulation experiments

These utilities make the simulation framework more user-friendly and help
streamline the process of setting up, running, and processing simulation results.
"""

from typing import Union
from cadCAD.configuration import Experiment  # type: ignore
from cadCAD.configuration.utils import config_sim  # type: ignore
from cadCAD.engine import ExecutionMode, ExecutionContext, Executor  # type: ignore
from cadCAD.tools.utils import add_parameter_labels # type: ignore
import pandas as pd
import sys
import os
from functools import partialmethod


class HiddenPrints:
    """
    Context manager to temporarily suppress stdout output.

    This class provides a way to silence print statements and progress bars from cadCAD and other libraries during simulation execution, making the output cleaner and more focused on relevant information.

    Attributes:
        is_active (bool): Whether output suppression is active
    """

    def __init__(self, is_active=True):
        """
        Initialize the context manager.

        Args:
            is_active (bool): If True, suppresses output; if False, allows normal output
        """
        self.is_active = is_active

    def __enter__(self):
        """
        Enter the context: redirect stdout to devnull if active.

        Also disables tqdm progress bars by modifying their initialization.
        """
        if self.is_active:
            # Save the original stdout and redirect to devnull
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
            from tqdm.auto import tqdm # type: ignore

            tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # type: ignore

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context: restore stdout if active.

        Also restores tqdm functionality to normal.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        if self.is_active:
            # Close the null file and restore original stdout
            sys.stdout.close()
            sys.stdout = self._original_stdout
            from tqdm.auto import tqdm

            # Restore tqdm functionality
            tqdm.__init__ = partialmethod(tqdm.__init__)  # type: ignore


def policy_aggregator(a, b):
    """
    Aggregate two policy functions or results.

    This function is used by cadCAD to combine the results of multiple policy
    functions within a single state update block. It handles None values appropriately.

    Args:
        a: First policy result (can be None)
        b: Second policy result (can be None)

    Returns:
        Combined policy result, or one of the inputs if the other is None
    """
    if a is None:
        return b
    elif b is None:
        return a
    else:
        return a + b


def sim_run(
    state_variables,
    params,
    psubs,
    N_timesteps,
    N_samples,
    use_label=False,
    assign_params: Union[bool, set] = True,
    drop_substeps=True,
    exec_mode="local",
    supress_cadCAD_print=False,
) -> pd.DataFrame:
    """
    Run cadCAD simulations without headaches.

    This function provides a simplified interface for configuring and executing
    cadCAD simulations. It handles common configuration options, executes the
    simulation, and formats the results as a pandas DataFrame.

    Args:
        state_variables (dict): Initial state variables
        params (list[dict]): List of parameter dictionaries for each simulation
        psubs (list[dict]): Partial state update blocks defining the model
        N_timesteps (int): Number of timesteps to simulate
        N_samples (int): Number of Monte Carlo runs per parameter set
        use_label (bool): Whether to add labels for substeps
        assign_params (Union[bool, set]): Whether to add parameter columns to output
        drop_substeps (bool): Whether to drop intermediate substeps
        exec_mode (str): Execution mode ('local' or 'single')
        supress_cadCAD_print (bool): Whether to suppress cadCAD output

    Returns:
        pd.DataFrame: Simulation results as a pandas DataFrame

    Example:
        ```python
        results = sim_run(
            state_variables=initial_state,
            params=[params_dict],
            psubs=MODEL_BLOCKS,
            N_timesteps=100,
            N_samples=5
        )
        ```
    """

    with HiddenPrints(is_active=supress_cadCAD_print):
        # Set-up sim_config
        simulation_parameters = {"N": N_samples, "T": range(N_timesteps), "M": params}
        sim_config = config_sim(simulation_parameters) # type: ignore

        # Create a new experiment
        exp = Experiment()
        exp.append_configs(
            sim_configs=sim_config,
            initial_state=state_variables,
            partial_state_update_blocks=psubs,
            policy_ops=[policy_aggregator],
        )
        configs = exp.configs

        # Set-up cadCAD executor
        if exec_mode == "local":
            _exec_mode = ExecutionMode().local_mode
        elif exec_mode == "single":
            _exec_mode = ExecutionMode().single_mode
        exec_context = ExecutionContext(
            _exec_mode, additional_objs={"deepcopy_off": True}
        )
        executor = Executor(
            exec_context=exec_context, configs=configs, supress_print=True
        )

        # Execute the cadCAD experiment
        (records, tensor_field, _) = executor.execute()

        # Parse the output as a pandas DataFrame
        df = pd.DataFrame(records) # type: ignore

        if drop_substeps == True:
            # Drop all intermediate substeps, keeping only initial state and final substep
            first_ind = (df.substep == 0) & (df.timestep == 0)
            last_ind = df.substep == max(df.substep)
            rows_to_keep = first_ind | last_ind
            df = df.loc[rows_to_keep].drop(columns=["substep"])

        if assign_params == False:
            pass
        else:
            # Add parameter labels to the dataframe for easier analysis
            df = add_parameter_labels(configs, df)

        # Based on Vitor Marthendal (@marthendalnunes) snippet
        if use_label == True:
            # Create a mapping from substep number to substep label
            psub_map = {
                order + 1: psub.get("label", "") for (order, psub) in enumerate(psubs)
            }
            psub_map[0] = "Initial State"
            df["substep_label"] = df.substep.map(psub_map)

        # Reset index for clean output
        df = df.reset_index(drop=False)
        return df
