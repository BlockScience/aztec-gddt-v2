"""
Command-line interface for running Aztec GDDT v2 simulations.

This module provides a command-line interface for executing experiments defined
in the Aztec GDDT v2 simulation package. It allows users to run pre-defined
simulation experiments with customizable parameters such as:

- Parallelization level
- Number of Monte Carlo runs
- Number of timesteps per simulation
- Parameter sweep samples
- Specific experiment selection

The module is executed when running the package directly with Python:
`python -m aztec_gddt [OPTIONS]`

Example:
    python -m aztec_gddt --experiment=fee_volatility --timesteps=1000 --mc_runs=5
"""

from aztec_gddt.experiment import test_run
from aztec_gddt.scenario_experiments import *
from datetime import datetime
import click
import logging
from pathlib import Path
from multiprocessing import cpu_count
from aztec_gddt import DEFAULT_LOGGER
import os
from aztec_gddt.analysis.metrics import *
from aztec_gddt.scenario_experiments import *
from aztec_gddt.analysis.execute import execute_sim, complexity_desc

# Set up the logger for this module
logger = logging.getLogger(DEFAULT_LOGGER)

# Define mapping of log level names to their corresponding logging constants
log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


@click.command()
@click.option('-z', '--parallelize', 'n_jobs',
              default=cpu_count())
@click.option('-s',
              '--sweep_samples',
              default=-1)
@click.option('-r',
              '--mc_runs',
              default=-1)
@click.option('-t',
              '--timesteps',
              default=-1)
@click.option('-e',
              '--experiment',
              default='test')
@click.option('-p',
              '--process',
              default=False,
              is_flag=True)
@click.option('-c',
              '--upload_to_cloud',
              default=False,
              is_flag=True)
@click.option('-a',
              '--alternate',
              default=False,
              is_flag=True)
@click.option(
    "-l",
    "--log-level",
    "log_level",
    type=click.Choice(list(log_levels.keys()), case_sensitive=False),
    default="info",
    help="Set the logging level.",
)
def main(process: bool,
         n_jobs: int,
         sweep_samples: int,
         mc_runs: int,
         timesteps: int,
         experiment: str,
         log_level: str,
         upload_to_cloud: bool,
         alternate: bool) -> None:

    logger.setLevel(log_levels[log_level])

    # Run all experiments or a specific one
    if experiment == 'all':
        # Execute all experiments defined in SCOPED_EXPERIMENTS
        for exp in SCOPED_EXPERIMENTS:
            run_exp(sweep_samples, mc_runs, timesteps, exp)
    else:
        # Find the requested experiment by name (case-insensitive)
        found_exps = [
            e for e in SCOPED_EXPERIMENTS if e.label.upper() == experiment.upper()]
        if len(found_exps) > 0:
            found_exp = found_exps[0]
        else:
            # Raise an exception if the experiment name is not recognized
            raise Exception(f"Experiment {experiment} not found!")

        # Run the found experiment
        run_exp(sweep_samples, mc_runs, timesteps, found_exp, alternate=alternate)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%SZ%z")


def run_exp(sweep_samples, mc_runs, timesteps, found_exp, alternate=True):
    """
    Execute a specific experiment with the provided configuration.

    This function applies the specified parameters to the experiment
    and then executes it using the execute_sim function.

    Args:
        sweep_samples: Number of parameter combinations to sample
        mc_runs: Number of Monte Carlo runs per parameter combination
        timesteps: Number of timesteps per simulation
        found_exp: The experiment object to execute
    """

    # Override experiment parameters if command-line values are provided
    if sweep_samples > 0:
        found_exp.N_config_sample = sweep_samples

    if timesteps > 0:
        found_exp.N_timesteps = timesteps

    if mc_runs > 0:
        found_exp.N_samples = mc_runs

    # Execute the simulation experiment
    execute_sim(found_exp, alternate=alternate)


# Standard Python idiom to check if this script is being run directly
if __name__ == "__main__":
    main()
