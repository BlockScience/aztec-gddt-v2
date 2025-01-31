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
         upload_to_cloud: bool) -> None:

    if upload_to_cloud:
        session = boto3.Session()
        s3 = session.client("s3")

    logger.setLevel(log_levels[log_level])


    if experiment == 'all':
        for exp in SCOPED_EXPERIMENTS:
            run_exp(sweep_samples, mc_runs, timesteps, exp)
    else:
        found_exps = [e for e in SCOPED_EXPERIMENTS if e.label.upper() == experiment.upper()]
        if len(found_exps) > 0:
            found_exp = found_exps[0]
        else:
            raise Exception(f"Experiment {experiment} not found!")
        
        run_exp(sweep_samples, mc_runs, timesteps, found_exp)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%SZ%z")

def run_exp(sweep_samples, mc_runs, timesteps, found_exp):
    if sweep_samples > 0:
        found_exp.N_config_sample = sweep_samples

    if timesteps > 0:
        found_exp.N_timesteps = timesteps

    if mc_runs > 0:
        found_exp.N_samples = mc_runs

    execute_sim(found_exp)



if __name__ == "__main__":
    main()
