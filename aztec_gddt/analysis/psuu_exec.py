"""
Parameter Selection under Uncertainty (PSUU) executor for Aztec GDDT v2.

This module handles:

1. Configuring and executing simulation experiments with parameter sweeps
2. Parallelizing simulations across multiple cores for performance
3. Saving and uploading simulation results to cloud storage
4. Calculating metrics and aggregating results for analysis

The main entry point is the `psuu` function, which orchestrates the entire simulation
workflow from experiment specification to results processing.
"""

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
from google.cloud import storage
from aztec_gddt.analysis.post_proc import post_process_sim_df


from aztec_gddt.default_params import DEFAULT_PARAMS, DEFAULT_INITIAL_STATE
from aztec_gddt.types import *
from aztec_gddt.helper_types import *
from aztec_gddt.structure import MODEL_BLOCKS
from aztec_gddt.analysis.metrics import *
import warnings

# Set up logging
logger = logging.getLogger('aztec-gddt-v2')

# Google Cloud Storage configuration
CLOUD_BUCKET_NAME = 'aztec-gddt-v2-sim'
CLOUD_PROJECT = 'aztec-bsci'


def psuu(
    exp_spec: ExperimentParamSpec,
    SWEEPS_PER_PROCESS: int = -1,
    PROCESSES: int = max(cpu_count()-1, 1),
    PARALLELIZE: bool = True,
    USE_JOBLIB: bool = True,
    RETURN_SIM_DF: bool = False,
    UPLOAD: bool = True,
    ignore_warnings: bool = True
):
    """
    Parameter Selection under Uncertainty (PSUU) function.

    This function orchestrates the execution of cadCAD simulations based on the
    provided experiment specification. It handles parameter sweep configuration,
    parallelization across multiple cores, results collection, post-processing,
    and optional cloud storage of results.

    Args:
        exp_spec (ExperimentParamSpec): Specification of the experiment parameters
        SWEEPS_PER_PROCESS (int): Number of parameter sweeps per process (-1 for auto)
        PROCESSES (int): Number of parallel processes to use
        PARALLELIZE (bool): Whether to parallelize the simulation
        USE_JOBLIB (bool): Whether to use joblib for parallelization
        RETURN_SIM_DF (bool): Whether to return the simulation DataFrame
        UPLOAD (bool): Whether to upload results to cloud storage
        ignore_warnings (bool): Whether to suppress warnings

    Returns:
        DataFrame or None: Simulation results if RETURN_SIM_DF is True, otherwise None

    Notes:
        - Results are saved to the 'data/runs/' directory with timestamped folders
        - When UPLOAD=True, results are also uploaded to Google Cloud Storage
    """
    # Suppress warnings if requested
    if ignore_warnings:
        warnings.filterwarnings('ignore')

    # Record the start time of the simulation run
    invoke_time = datetime.now()
    logger.info(f"{exp_spec.label} Run invoked at {invoke_time}")

    # Extract key experiment parameters
    TIMESTEPS = exp_spec.N_timesteps

    # Make a deep copy of default parameters to avoid modifying the original
    default_params = deepcopy(DEFAULT_PARAMS)

    # Create a sweep of all parameter combinations
    sweep_params = sweep_cartesian_product(
        {
            **{k: [v] for k, v in default_params.items()},
            **exp_spec.params_swept_env,
            **exp_spec.params_swept_control,
        }
    )

    # Sample the sweep space if a sample size is specified
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

    # Define which parameters to include in the output dataframe
    assign_params = {
        *list(exp_spec.params_swept_control.keys()),
        *list(exp_spec.params_swept_env.keys()),
        *['MINIMUM_MULTIPLIER_CONGESTION',
          'PROVING_COST_MODIFICATION_E',
            'FEE_JUICE_PRICE_MODIFICATION_E',
            'RELATIVE_TARGET_MANA_PER_BLOCK',
            'MAXIMUM_MANA_PER_BLOCK',
            'market_price_eth']
    }

    # Calculate the dimensions of the simulation for logging
    sweep_combinations = len(sweep_params['label'])
    n_sweeps = exp_spec.N_config_sample if exp_spec.N_config_sample > 0 else sweep_combinations
    N_measurements = n_sweeps * TIMESTEPS * exp_spec.N_samples
    traj_combinations = n_sweeps * exp_spec.N_samples

    # Log simulation dimensions
    logger.info(
        f"{exp_spec.label} dimensions: N_jobs={PROCESSES:,}, N_t={TIMESTEPS:,}, N_sweeps={n_sweeps:,}, N_mc={exp_spec.N_samples:,}, N_trajectories={traj_combinations:,}, N_measurements={N_measurements:,}")

    # Extract parallelization options
    parallelize = PARALLELIZE
    use_joblib = USE_JOBLIB

    # Record simulation start time
    sim_start_time = datetime.now()
    logger.info(
        f"{exp_spec.label} starting at {sim_start_time}, ({sim_start_time - invoke_time} since invoke)")

    # Single-process execution path
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

        # Apply post-processing to add derived metrics
        sim_df = post_process_sim_df(sim_df)

    # Parallelized execution path
    else:
        # Determine sweeps per process either from parameter or automatically
        if SWEEPS_PER_PROCESS > 0:
            sweeps_per_process = SWEEPS_PER_PROCESS
        else:
            # Auto-calculate a reasonable number of sweeps per process
            sweeps_per_process = max(
                min(int(traj_combinations / PROCESSES), 30), 1)
        processes = PROCESSES

        # Split the parameter space into chunks for parallel processing
        chunk_size = sweeps_per_process
        split_dicts = [
            {k: v[i: i + chunk_size] for k, v in sweep_params_samples.items()}
            for i in range(0, len(list(sweep_params_samples.values())[0]), chunk_size)
            for j in range(exp_spec.N_samples)
        ]

        # Set up output directories for results
        sim_folder_path = Path(f"data/runs/")
        base_folder = Path(
            f"{exp_spec.label if exp_spec.label != '' else 'undefined'}/{datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')}")
        output_folder_path = sim_folder_path / base_folder
        output_folder_path.mkdir(parents=True, exist_ok=True)

        # Save experiment specification to JSON file
        with open(output_folder_path / "spec.json", "w") as fid:
            fid.write(exp_spec.to_json())
        output_path = str(output_folder_path / "timestep_tensor")

        # Upload spec.json to cloud storage if requested
        if UPLOAD:
            storage_client = storage.Client(project=CLOUD_PROJECT)
            bucket = storage_client.bucket(CLOUD_BUCKET_NAME)
            blob = bucket.blob(str(base_folder / "spec.json"))  # type: ignore
            blob.upload_from_filename(
                output_folder_path / "spec.json")  # type: ignore

        # Define function to run a single chunk of the simulation
        def run_chunk(i_chunk, sweep_params, pickle_file=True, upload=UPLOAD, post_process=True):
            """
            Run a single chunk of the simulation with specific parameters.

            Args:
                i_chunk (int): Chunk identifier
                sweep_params (dict): Parameter sweep dictionary for this chunk
                pickle_file (bool): Whether to save results as pickle files
                upload (bool): Whether to upload results to cloud storage
                post_process (bool): Whether to calculate derived metrics

            Returns:
                None: Results are saved to files and/or uploaded to cloud
            """
            # Set up simulation arguments for this chunk
            sim_args = (
                DEFAULT_INITIAL_STATE,
                sweep_params,
                MODEL_BLOCKS,
                TIMESTEPS,
                1,
            )
            # Run simulation
            sim_df = easy_run(
                *sim_args,
                exec_mode="single",
                assign_params=assign_params,
                deepcopy_off=True,
                supress_print=True
            )

            # Add chunk identifiers to the results
            sim_df["simulation"] = i_chunk
            sim_df["subset"] = i_chunk * SWEEPS_PER_PROCESS + sim_df["subset"]

            # Apply post-processing to add derived metrics
            sim_df = post_process_sim_df(sim_df)
            output_filename = output_path + f"-{i_chunk}.pkl.gz"

            # Save results to pickle file if requested
            if pickle_file or upload:
                sim_df.to_pickle(output_filename)

            # Upload results to cloud storage if requested
            if upload:
                storage_client = storage.Client(project=CLOUD_PROJECT)
                bucket = storage_client.bucket(CLOUD_BUCKET_NAME)
                blob = bucket.blob(str(base_folder /
                                   # type: ignore
                                       f"timestep_tensor-{i_chunk}.pkl.gz"))
                blob.upload_from_filename(str(output_filename))  # type: ignore
                os.remove(str(output_filename))

            # Generate and save trajectory-level metrics if requested
            if post_process:
                # Calculate metrics across trajectories
                agg_df, c_agg_df = retrieve_feature_df(
                    sim_df,
                    list(exp_spec.params_swept_control.keys()),
                    exp_spec.relevant_per_trajectory_group_metrics)

                # Define output filename for trajectory-level metrics
                agg_output_filename = output_folder_path / \
                    f"trajectory_tensor-{i_chunk}.pkl.gz"

                # Save trajectory metrics to pickle file if requested
                if pickle_file:
                    agg_df.to_pickle(agg_output_filename)
                    # Upload trajectory metrics to cloud storage if requested
                    if upload:
                        blob = bucket.blob(
                            # type: ignore
                            str(base_folder / f"trajectory_tensor-{i_chunk}.pkl.gz"))
                        blob.upload_from_filename(
                            str(agg_output_filename))  # type: ignore

        # Prepare arguments for parallel execution
        args = enumerate(split_dicts)

        # Use joblib for parallel execution if requested
        if use_joblib:
            Parallel(n_jobs=processes)(
                delayed(run_chunk)(i_chunk, sweep_params)
                for (i_chunk, sweep_params) in tqdm(args, desc='Simulation Chunks', total=len(split_dicts))
            )
        # Otherwise run chunks sequentially
        else:
            for i_chunk, sweep_params in tqdm(args):
                run_chunk(i_chunk, sweep_params)

        # Load and combine all simulation results if requested
        if RETURN_SIM_DF:
            print(output_path)
            sim_df = pd.concat(
                [pd.read_pickle(part, compression="gzip")
                 for part in glob(output_path+"*")]
            )

    # Record end time and calculate performance metrics
    end_start_time = datetime.now()
    duration: float = (end_start_time - sim_start_time).total_seconds()

    # Log performance metrics
    logger.info(
        f"{exp_spec.label} Run finished at {end_start_time}, ({end_start_time - sim_start_time} since sim start)")
    logger.info(
        f"{exp_spec.label} Run Performance Numbers; Duration (s): {duration:,.2f}, Measurements Per Second: {N_measurements/duration:,.2f} M/s, Measurements per Job * Second: {N_measurements/(duration * PROCESSES):,.2f} M/(J*s), Jobs * Seconds per Trajectory : {duration * PROCESSES / traj_combinations:,.2f}")

    # Return simulation DataFrame if requested
    if RETURN_SIM_DF:
        return sim_df  # type: ignore
    else:
        pass

    # Aggregate trajectory metrics if using joblib
    if use_joblib:
        # type: ignore
        files = glob(str(output_folder_path / f"trajectory_tensor-*.pkl.gz"))
        dfs = []
        # Load each file and reset index
        for file in files:
            dfs.append(pd.read_pickle(file).reset_index())
        # Concatenate all trajectory metric DataFrames
        agg_df = pd.concat(dfs)

        # Save aggregated trajectory metrics to CSV and pickle files
        agg_df.to_csv(str(output_folder_path / f"trajectory_tensor.csv.gz"))
        agg_df.to_pickle(str(output_folder_path / f"trajectory_tensor.pkl.gz"))

        # Upload aggregated trajectory metrics to cloud storage if requested
        if UPLOAD:
            storage_client = storage.Client(project=CLOUD_PROJECT)
            bucket = storage_client.bucket(CLOUD_BUCKET_NAME)
            # Upload CSV version
            blob = bucket.blob(
                str(base_folder / f"trajectory_tensor.csv.gz"))  # type: ignore
            blob.upload_from_filename(
                # type: ignore
                str(output_folder_path / f"trajectory_tensor.csv.gz"))
            blob = bucket.blob(
                str(base_folder / f"trajectory_tensor.pkl.gz"))  # type: ignore
            blob.upload_from_filename(
                # type: ignore
                str(output_folder_path / f"trajectory_tensor.pkl.gz"))
    return None
