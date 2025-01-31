from aztec_gddt.experiment import test_run
from aztec_gddt.scenario_experiments import *
from datetime import datetime
import click
import logging
from pathlib import Path
from multiprocessing import cpu_count
from aztec_gddt import DEFAULT_LOGGER
import boto3  # type: ignore
import os
from aztec_gddt.analysis.metrics import *
from aztec_gddt.scenario_experiments import *
from aztec_gddt.analysis.execute import execute_sim, complexity_desc


logger = logging.getLogger(DEFAULT_LOGGER)
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
              default=5)
@click.option('-t',
              '--timesteps',
              default=1_000)
@click.option('-p',
              '--process',
              default=False,
              is_flag=True)
@click.option('-c',
              '--upload_to_cloud',
              default=False,
              is_flag=True)
@click.option('--no_parallelize',
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
         log_level: str,
         upload_to_cloud: bool,
         no_parallelize: bool) -> None:

    if upload_to_cloud:
        session = boto3.Session()
        s3 = session.client("s3")

    logger.setLevel(log_levels[log_level])

    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%SZ%z")

    # test_run()
    from aztec_gddt.helper_types import ExperimentParamSpec

    exp_spec = ExperimentParamSpec(
        label='test',
        params_swept_control={
            'RELATIVE_TARGET_MANA_PER_BLOCK': [0.50, 0.90],
            'MAXIMUM_MANA_PER_BLOCK': [20_000_000, 40_000_000],
        },
        params_swept_env={
            'SEQUENCER_L1_GAS_PRICE_THRESHOLD_E': [100, 1_000],
            'TOTAL_MANA_MULTIPLIER_E': [1.0, 10.0]
        },
        N_timesteps=500,
        N_samples=1,
        N_config_sample=-1,
        relevant_per_trajectory_metrics=list(
            PER_TRAJECTORY_METRICS_LABELS.keys()),
        relevant_per_trajectory_group_metrics=list(
            PER_TRAJECTORY_GROUP_METRICS_LABELS.keys()),
    )
    CONTROL_PARAMS = list(exp_spec.params_swept_control.keys())
    sim_df, exec_time = execute_sim(exp_spec)
    agg_df, c_agg_df = retrieve_feature_df(
        sim_df, CONTROL_PARAMS, exp_spec.relevant_per_trajectory_group_metrics)


if __name__ == "__main__":
    main()
