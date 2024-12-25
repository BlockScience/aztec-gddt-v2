from aztec_gddt.experiment import test_run
from datetime import datetime
import click
import logging
from pathlib import Path
from multiprocessing import cpu_count
from aztec_gddt import DEFAULT_LOGGER
import boto3 # type: ignore
import os

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

    test_run()

if __name__ == "__main__":
    main()
