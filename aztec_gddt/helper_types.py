"""
Helper types and experiment wrappers for Aztec GDDT simulations.

This module provides dataclasses and utilities for organizing, configuring, and executing simulation experiments. It includes:

1. ExperimentWrapper - A wrapper for cadCAD experiment configurations
2. ExperimentParamSpec - A specification for parameter sweeps and experiment configuration

These classes facilitate parameter sweeping, experiment setup, and distribution of simulation workloads across multiple processes. They bridge between the experiment definitions in scenario_experiments.py and the simulation execution in analysis/execute.py.
"""

from aztec_gddt.types import *
from aztec_gddt.default_params import *
from aztec_gddt.structure import MODEL_BLOCKS
from aztec_gddt.utils import policy_aggregator
from cadCAD.configuration import Experiment  # type: ignore
from cadCAD.configuration.utils import config_sim  # type: ignore
from cadCAD.tools.preparation import sweep_cartesian_product  # type: ignore
from random import sample
from dataclasses import dataclass, field
import numpy as np
from typing import Iterable
from dataclasses_json import dataclass_json


@dataclass
class ExperimentWrapper():
    """
    Wrapper class for cadCAD experiment configurations.

    This class encapsulates a cadCAD experiment along with metadata about the experiment's parameters and scale. It provides methods for distributing the experiment across multiple processes.

    Attributes:
        params_swept_control (dict): Dictionary of control parameters being swept
        params_swept_env (dict): Dictionary of environmental parameters being swept
        N_timesteps (int): Number of timesteps per simulation
        N_samples (int): Number of Monte Carlo runs per parameter combination
        N_configs (int): Number of parameter configurations being tested
        experiment (Experiment): The cadCAD experiment object
        label (str): Descriptive label for the experiment
    """
    params_swept_control: dict
    params_swept_env: dict
    N_timesteps: int
    N_samples: int
    N_configs: int
    experiment: Experiment
    label: str = ""

    def split_into_chunks(self, N_chunks) -> Iterable[Experiment]:
        """
        Split the experiment into multiple chunks for parallel execution.

        This method divides the experiment's configuration space into approximately equal-sized chunks to distribute the computational load across multiple processes.

        Args:
            N_chunks (int): Number of chunks to split the experiment into

        Yields:
            Experiment: A cadCAD experiment object containing a subset of configurations

        Note:
            This is used for parallelizing experiment execution across multiple CPU cores or compute nodes.
        """
        # Split the configuration list into N_chunks approximately equal parts
        splitted_configs: list[dict] = list(np.array_split(
            self.experiment.configs, N_chunks))  # type: ignore

        # Create a new experiment for each chunk
        for i, splitted_config in enumerate(splitted_configs):
            exp = Experiment()
            exp.configs = list(splitted_configs)
            yield exp


@dataclass_json
@dataclass
class ExperimentParamSpec():
    """
    Specification for experiment parameters and configuration.

    This class defines the parameters for a simulation experiment, including which parameters to sweep, how many timesteps and Monte Carlo runs to perform, and which metrics to track. It provides methods to prepare and execute the experiment.

    Attributes:
        params_swept_control (dict): Control parameters to sweep (parameter name -> list of values)
        params_swept_env (dict): Environmental parameters to sweep (parameter name -> list of values)
        N_timesteps (int): Number of timesteps per simulation
        N_samples (int): Number of Monte Carlo runs per parameter combination
        N_config_sample (int): Number of parameter configurations to sample (if >0)
        relevant_per_trajectory_metrics (list[str]): Metrics to track per trajectory
        relevant_per_trajectory_group_metrics (list[str]): Group metrics to track
        label (str): Descriptive label for the experiment
    """
    params_swept_control: dict
    params_swept_env: dict
    N_timesteps: int
    N_samples: int
    N_config_sample: int
    relevant_per_trajectory_metrics: list[str] = field(default_factory=list)
    relevant_per_trajectory_group_metrics: list[str] = field(
        default_factory=list)
    label: str = ""

    @property
    def N_params(self) -> int:
        """
        Calculate the total number of parameter combinations in the sweep.

        This property computes the product of the number of values for each swept parameter, giving the total size of the parameter space.

        Returns:
            int: Total number of parameter combinations
        """
        N_params = 1
        # Multiply by the number of values for each control parameter
        for k, v in self.params_swept_control.items():
            N_params *= len(v)
        # Multiply by the number of values for each environmental parameter
        for k, v in self.params_swept_env.items():
            N_params *= len(v)
        return N_params

    @property
    def N_trajectories(self) -> int:
        """
        Calculate the total number of simulation trajectories.

        This property computes the product of the number of parameter combinations and the number of Monte Carlo runs, giving the total number of simulation trajectories that will be executed.

        Returns:
            int: Total number of simulation trajectories
        """
        return self.N_params * self.N_samples

    @property
    def N_measurements(self) -> int:
        """
        Calculate the total number of measurements across all simulations.

        This property computes the product of the number of trajectories and the number of timesteps, giving the total number of state measurements that will be collected.

        Returns:
            int: Total number of measurements
        """
        return self.N_trajectories * self.N_timesteps

    def print_control_params(self):
        """
        Print the control parameters being swept in this experiment.

        This method outputs all control parameters and their values to the console, useful for debugging and logging experiment configurations.
        """
        for k, v in self.params_swept_control.items():
            print(f"{k}: {v}")

    def print_env_params(self):
        """
        Print the environmental parameters being swept in this experiment.

        This method outputs all environmental parameters and their values to the console, useful for debugging and logging experiment configurations.
        """
        for k, v in self.params_swept_env.items():
            print(f"{k}: {v}")

    def prepare(self=1) -> ExperimentWrapper:
        """
        Prepare the experiment for execution.

        This method creates a cadCAD experiment configuration based on the parameters specified in this ExperimentParamSpec. It combines default parameters with the swept parameters, configures the simulation, and returns an ExperimentWrapper
        that can be executed.

        Returns:
            ExperimentWrapper: A wrapper object containing the prepared experiment

        Note:
            If N_config_sample > 0, a random subset of the parameter space will be sampled rather than the full cartesian product.
        """
        # Start with default parameters as single-item lists
        default_params = {k: [v] for k, v in DEFAULT_PARAMS.items()}
        default_params['N_timesteps'] = [self.N_timesteps] 

        # Combine default parameters with swept parameters
        params_to_sweep = {**default_params, **
                           self.params_swept_env, **self.params_swept_control}

        # Generate the cartesian product of all parameter values
        prepared_params = sweep_cartesian_product(params_to_sweep)

        # Set up initial states
        states_list = [DEFAULT_INITIAL_STATE]

        # Create and configure the cadCAD experiment
        exp = Experiment()
        for state in states_list:
            simulation_parameters = {"N": self.N_samples,
                                     "T": range(self.N_timesteps), "M": prepared_params}
            sim_config = config_sim(simulation_parameters)  # type: ignore
            exp.append_configs(
                sim_configs=sim_config,
                initial_state=state,
                partial_state_update_blocks=MODEL_BLOCKS,
                policy_ops=[policy_aggregator],
            )

        # If sampling is enabled, select a random subset of configurations
        if int(self.N_config_sample) > 0:
            exp.configs = sample(exp.configs, int(self.N_config_sample))

        # Count the final number of configurations
        N_configs = len(exp.configs)

        # Create and return the experiment wrapper
        wrapper = ExperimentWrapper(params_swept_control=self.params_swept_env,
                                    params_swept_env=self.params_swept_env,
                                    N_timesteps=self.N_timesteps,
                                    N_samples=self.N_samples,
                                    N_configs=N_configs,
                                    experiment=exp,
                                    label=self.label)
        return wrapper
