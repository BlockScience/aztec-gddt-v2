from copy import deepcopy

import numpy as np
import pandas as pd
from cadCAD.tools import easy_run  # type: ignore
from cadCAD.tools.preparation import sweep_cartesian_product  # type: ignore
from pandas import DataFrame
from random import sample
from datetime import datetime
from joblib import Parallel, delayed  # type: ignore
from glob import glob
import re
from tqdm.auto import tqdm  # type: ignore
import logging
from pathlib import Path
import os
from multiprocessing import cpu_count
import boto3  # type: ignore

from aztec_gddt.default_params import DEFAULT_PARAMS, DEFAULT_INITIAL_STATE
from aztec_gddt.types import *
from aztec_gddt.helper_types import *
from aztec_gddt.structure import MODEL_BLOCKS
from aztec_gddt.analysis.metrics import *
import warnings

logger = logging.getLogger('aztec-gddt-v2')
CLOUD_BUCKET_NAME = 'aztec-gddt-v2-sim'


def psuu(
    exp_spec: ExperimentParamSpec,
    SWEEPS_PER_PROCESS: int = 20,
    PROCESSES: int = cpu_count(),
    PARALLELIZE: bool = True,
    USE_JOBLIB: bool = False,
    RETURN_SIM_DF: bool = False,
    UPLOAD_TO_S3: bool = False,
    ignore_warnings: bool = True
):
    """Function which runs the cadCAD simulations

    Returns:
        DataFrame: A dataframe of simulation data
    """
    if ignore_warnings:
        warnings.filterwarnings('ignore')

    invoke_time = datetime.now()
    logger.info(f"{{exp_spec.label}} Run invoked at {invoke_time}")

    TIMESTEPS = exp_spec.N_timesteps

    default_params = deepcopy(DEFAULT_PARAMS)

    sweep_params = sweep_cartesian_product(
        {
            **{k: [v] for k, v in default_params.items()},
            **exp_spec.params_swept_env,
            **exp_spec.params_swept_control,
        }
    )

    # Sample the sweep space
    sweep_params_samples = {
        k: sample(
            v, exp_spec.N_config_sample) if exp_spec.N_config_sample > 0 else v
        for k, v in sweep_params.items()
    }

    # Load simulation arguments
    sim_args = (
        DEFAULT_INITIAL_STATE,
        sweep_params_samples,
        MODEL_BLOCKS,
        TIMESTEPS,
        exp_spec.N_samples,
    )
    assign_params = {
        *list(exp_spec.params_swept_control.keys()),
        *list(exp_spec.params_swept_env.keys()),
        *['MINIMUM_MULTIPLIER_CONGESTION',
          'PROVING_COST_MODIFICATION_E',
            'FEE_JUICE_PRICE_MODIFICATION_E',
            'RELATIVE_TARGET_MANA_PER_BLOCK',
            'MAXIMUM_MANA_PER_BLOCK']
    }

    sweep_combinations = len(sweep_params['label'])

    n_sweeps = exp_spec.N_config_sample if exp_spec.N_config_sample > 0 else sweep_combinations
    N_measurements = n_sweeps * TIMESTEPS * exp_spec.N_samples

    traj_combinations = n_sweeps * exp_spec.N_samples

    logger.info(
        f"{exp_spec.label} run Dimensions: N_jobs={PROCESSES=:,}, N_t={TIMESTEPS=:,}, N_sweeps={n_sweeps:,}, N_mc={exp_spec.N_samples:,}, N_trajectories={traj_combinations:,}, N_measurements={N_measurements:,}")

    parallelize = PARALLELIZE
    use_joblib = USE_JOBLIB

    sim_start_time = datetime.now()
    logger.info(
        f"{exp_spec.label} Exploratory Run starting at {sim_start_time}, ({sim_start_time - invoke_time} since invoke)")
    if parallelize is False:
        # Load simulation arguments
        sim_args = (
            DEFAULT_INITIAL_STATE,
            sweep_params_samples,
            MODEL_BLOCKS,
            TIMESTEPS,
            exp_spec.N_samples,
        )
        # Run simulation and write results to disk
        sim_df = easy_run(
            *sim_args,
            exec_mode="single",
            assign_params=assign_params,
            deepcopy_off=True,
            supress_print=True
        )
    else:
        sweeps_per_process = SWEEPS_PER_PROCESS
        processes = PROCESSES

        chunk_size = sweeps_per_process
        split_dicts = [
            {k: v[i: i + chunk_size] for k, v in sweep_params_samples.items()}
            for i in range(0, len(list(sweep_params_samples.values())[0]), chunk_size)
        ]
        sim_folder_path = Path(f"data/runs/{exp_spec.label if exp_spec.label != '' else 'undefined'}/")
        base_folder = Path(
            f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')}")
        output_folder_path = sim_folder_path / base_folder
        output_folder_path.mkdir(parents=True, exist_ok=True)

        with open(output_folder_path / "spec.json", "w") as fid:
            fid.write(exp_spec.to_json())
        output_path = str(output_folder_path / "timestep_tensor")

        def run_chunk(i_chunk, sweep_params, pickle_file=True, upload_to_s3=UPLOAD_TO_S3, post_process=True):
            logger.debug(f"{i_chunk}, {datetime.now()}")
            sim_args = (
                DEFAULT_INITIAL_STATE,
                sweep_params,
                MODEL_BLOCKS,
                TIMESTEPS,
                exp_spec.N_samples,
            )
            # Run simulationz
            sim_df = easy_run(
                *sim_args,
                exec_mode="single",
                assign_params=assign_params,
                deepcopy_off=True,
                supress_print=True
            )

            if upload_to_s3:
                session = boto3.Session()
                s3 = session.client("s3")

            sim_df["subset"] = i_chunk * SWEEPS_PER_PROCESS + sim_df["subset"]
            output_filename = output_path + f"-{i_chunk}.pkl.gz"

            if pickle_file or upload_to_s3:
                sim_df.to_pickle(output_filename)
            if upload_to_s3:
                s3.upload_file(str(output_filename),
                               CLOUD_BUCKET_NAME,
                               str(base_folder /
                                   f"timestep_tensor-{i_chunk}.pkl.gz")
                               )
                os.remove(str(output_filename))

            if post_process:
                sim_df['normed_congestion_multiplier'] = sim_df['congestion_multiplier'] / \
                    sim_df['MINIMUM_MULTIPLIER_CONGESTION']
                sim_df['average_mana_per_block_per_target'] = sim_df.apply(lambda df: sum(b.tx_total_mana for b in df.last_epoch.slots) / len(df.last_epoch.slots) / (
                    df.MAXIMUM_MANA_PER_BLOCK * df.RELATIVE_TARGET_MANA_PER_BLOCK) if len(df.last_epoch.slots) > 0 else float('nan'), axis='columns')
                sim_df['average_mana_per_block_per_max'] = sim_df.apply(lambda df: sum(b.tx_total_mana for b in df.last_epoch.slots) / len(
                    df.last_epoch.slots) / (df.MAXIMUM_MANA_PER_BLOCK) if (len(df.last_epoch.slots) > 0 & df.last_epoch.finalized) else float('nan'), axis='columns')

                agg_df, c_agg_df = retrieve_feature_df(
                    sim_df, 
                    list(exp_spec.params_swept_control.keys()), 
                    exp_spec.relevant_per_trajectory_group_metrics)

                agg_output_filename = output_folder_path / \
                    f"trajectory_tensor-{i_chunk}.csv.gz"
                
                c_agg_output_filename = output_folder_path / \
                    f"trajectory_bool_tensor-{i_chunk}.csv.gz"
                
                if pickle_file:
                    agg_df.to_csv(agg_output_filename)
                    c_agg_df.to_csv(agg_output_filename)
                    if upload_to_s3:
                        s3.upload_file(str(agg_output_filename),
                                       CLOUD_BUCKET_NAME,
                                       str(base_folder / f"trajectory_tensor-{i_chunk}.csv.gz"))
                        s3.upload_file(str(c_agg_output_filename),
                                       CLOUD_BUCKET_NAME,
                                       str(base_folder / f"trajectory_bool_tensor-{i_chunk}.csv.gz"))
        args = enumerate(split_dicts)
        if use_joblib:
            Parallel(n_jobs=processes)(
                delayed(run_chunk)(i_chunk, sweep_params)
                for (i_chunk, sweep_params) in tqdm(args, desc='Simulation Chunks', total=len(split_dicts))
            )
        else:
            for i_chunk, sweep_params in tqdm(args):
                run_chunk(i_chunk, sweep_params)

        if RETURN_SIM_DF:
            sim_df = pd.concat(
                [pd.read_pickle(part, compression="gzip")
                 for part in glob(output_path+"*")]
            )

    end_start_time = datetime.now()
    duration: float = (end_start_time - sim_start_time).total_seconds()
    logger.info(
        f"{exp_spec.label} Run finished at {end_start_time}, ({end_start_time - sim_start_time} since sim start)")
    logger.info(
        f"{exp_spec.label} Run Performance Numbers; Duration (s): {duration:,.2f}, Measurements Per Second: {N_measurements/duration:,.2f} M/s, Measurements per Job * Second: {N_measurements/(duration * PROCESSES):,.2f} M/(J*s)")
    if RETURN_SIM_DF:
        return sim_df  # type: ignore
    else:
        pass

    if use_joblib and UPLOAD_TO_S3:
        files = glob(str(output_folder_path / f"trajectory_tensor-*.pkl.gz"))
        dfs = []
        for file in files:
            dfs.append(pd.read_pickle(file).reset_index())
        agg_df = pd.concat(dfs)
        agg_df.to_pickle(str(output_folder_path / f"trajectory_tensor.pkl.gz"))
        session = boto3.Session()
        s3 = session.client("s3")
        s3.upload_file(str(output_folder_path / f"trajectory_tensor.pkl.gz"),
                       CLOUD_BUCKET_NAME,
                       str(base_folder / f"trajectory_tensor.pkl.gz"))
    return None
