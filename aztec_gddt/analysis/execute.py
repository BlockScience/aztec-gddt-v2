from dataclasses import dataclass
import pandas as pd
from aztec_gddt.helper_types import ExperimentWrapper

@dataclass
class ExecutionTime():
    before_setup: float = float('nan')
    before_run: float = float('nan')
    after_run: float = float('nan')
    after_proc: float = float('nan')

    @property
    def simulation(this):
        return this.after_run - this.before_run
    
    @property
    def workflow(this):
        return this.after_proc - this.before_run


def execute_sim(exp_wrapper: ExperimentWrapper) -> tuple[pd.DataFrame, ExecutionTime]:
    from time import time

    exec_time = ExecutionTime()

    exec_time.before_run = time()
    exp = exp_wrapper.experiment


    from cadCAD.engine import ExecutionContext, ExecutionMode, Executor


    _exec_mode = ExecutionMode().single_mode
    exec_context = ExecutionContext(_exec_mode, additional_objs={'deepcopy_off': True})
    executor = Executor(exec_context=exec_context, configs=exp.configs, supress_print=False)

    # Execute the cadCAD experiment
    exec_time.before_run = time()
    (records, tensor_field, _) = executor.execute()
    exec_time.after_run = time()

    # Parse the output as a pandas DataFrame
    df = pd.DataFrame(records) # type: ignore

    # Drop substeps
    first_ind = (df.substep == 0) & (df.timestep == 0)
    last_ind = df.substep == max(df.substep)
    inds_to_drop = first_ind | last_ind
    df = df.loc[inds_to_drop].drop(columns=['substep'])


    # Assign Params
    M_dict = exp.configs[0].sim_config['M']
    params_set = set(M_dict.keys())

    selected_params = params_set
    # Attribute parameters to each row*

    from cadCAD.tools.execution.easy_run import select_config_M_dict # type: ignore
    params_dict = select_config_M_dict(exp.configs, 0, selected_params)

    # Handles all cases of parameter types including list
    for key, value in params_dict.items():
        df[key] = df.apply(lambda _: value, axis=1)

    for i, (_, n_df) in enumerate(df.groupby(['simulation', 'subset', 'run'])):
        params_dict = select_config_M_dict(exp.configs, i, selected_params)
        for key, value in params_dict.items():
            df.loc[n_df.index, key] = df.loc[n_df.index].apply(
                lambda _: value, axis=1)
            

    sim_df = df
    exec_time.after_proc = time()


    # Post Processing Metrics
    sim_df['normed_congestion_multiplier'] = sim_df['congestion_multiplier'] / sim_df['MINIMUM_MULTIPLIER_CONGESTION']
    sim_df['average_mana_per_block_per_target'] = sim_df.apply(lambda df: sum(b.tx_total_mana for b in df.last_epoch.slots) / len(df.last_epoch.slots) / (df.MAXIMUM_MANA_PER_BLOCK * df.RELATIVE_TARGET_MANA_PER_BLOCK) if len(df.last_epoch.slots) > 0 else float('nan'), axis='columns')
    sim_df['average_mana_per_block_per_max'] = sim_df.apply(lambda df: sum(b.tx_total_mana for b in df.last_epoch.slots) / len(df.last_epoch.slots) / (df.MAXIMUM_MANA_PER_BLOCK) if (len(df.last_epoch.slots) > 0 & df.last_epoch.finalized) else float('nan'), axis='columns')
    return sim_df, exec_time

def complexity_desc(sim_df: pd.DataFrame, exec_time: ExecutionTime) -> str:
    N_trajectories = len(sim_df[['subset', 'run']].drop_duplicates())

    text = f"""
    #### Computational Complexity:
    1. Total number of parameter combinations: {len(sim_df.subset.unique()):,}
    2. Total number of Monte Carlo runs per parameter combination: {len(sim_df.run.unique()):,}
    3. Total number of trajectories: {N_trajectories:,}
    4. Total number of timesteps per trajectory: {sim_df.timestep.max():,}
    5. Total number of state measurements: {len(sim_df):,}
    6. Workflow execution time: {exec_time.workflow:,.3} seconds ({exec_time.workflow / N_trajectories:,.3} seconds per trajectory)
    7. Engine execution time: {exec_time.simulation:,.3} seconds ({exec_time.simulation / N_trajectories:,.3} seconds per trajectory)
    """

    return text