"""
Analysis metrics for evaluating Aztec GDDT v2 simulation results.

This module provides metrics functions for analyzing and evaluating simulation
trajectories from the Aztec GDDT v2 simulation. It includes:

1. Per-trajectory metrics (T-M*) - Calculate various properties of individual simulation trajectories
2. Group trajectory metrics (TG-M*) - Aggregate metrics across trajectory groups
3. Helper functions for computing aggregate statistics
4. Tools for feature extraction and metric collapsing
"""

import pandas as pd
import numpy as np
from aztec_gddt.types import *


def relative_volatility_across_time(traj_df: pd.DataFrame) -> float:
    """
    Calculate the relative volatility between market price and base fee.

    Average relative volatility measures the impact of an "outside" environmental variable on one or more system variables by comparing the volatility of the environmental variable to the volatility of the system variable. This may be thought of as "inputting" the environmental variable's time series into the network and measuring the resulting "output" volatility of the system variable's time series. Relative volatility is used extensively in engineering to measure system response to input 'signals', making this a useful metric for such signal processing.

    Args:
        traj_df (pd.DataFrame): Trajectory dataframe with simulation results

    Returns:
        float: Ratio of market price standard deviation to base fee standard deviation
    """
    return traj_df.market_price_juice_per_gwei.std(skipna=True) / traj_df.base_fee.std(skipna=True)


def empty_blocks_during_trajectory(traj_df: pd.DataFrame) -> int:
    """
    Count the number of empty blocks (blocks with no transactions) in the trajectory.

    Empty blocks indicate inefficiency in the system as resources are spent on
    blocks that don't process any transactions.

    Args:
        traj_df (pd.DataFrame): Trajectory dataframe with simulation results

    Returns:
        int: Total number of empty blocks
    """
    return traj_df.iloc[-1].cumm_empty_blocks


def unproven_epochs_during_trajectory(traj_df: pd.DataFrame) -> int:
    """
    Count the number of epochs that didn't get proven during the trajectory.

    Unproven epochs indicate issues with the proving incentives or parametrization,
    which impacts the security and finality guarantees of the system.

    Args:
        traj_df (pd.DataFrame): Trajectory dataframe with simulation results

    Returns:
        int: Total number of unproven epochs
    """
    return traj_df.iloc[-1].cumm_unproven_epochs


def fraction_dropped_tx_during_trajectory(traj_df: pd.DataFrame) -> float:
    """
    Calculate the fraction of transactions that were dropped during the trajectory.

    Dropped transactions indicate that users' transactions failed to be included,
    which is a measure of system reliability from a user perspective.

    Args:
        traj_df (pd.DataFrame): Trajectory dataframe with simulation results

    Returns:
        float: Fraction of dropped transactions (0.0 to 1.0)
    """
    return traj_df.iloc[-1].cumm_dropped_tx / traj_df.iloc[-1].cumm_total_tx


def fraction_excluded_tx_during_trajectory(traj_df: pd.DataFrame) -> float:
    """
    Calculate the fraction of transactions that were excluded during the trajectory.

    Excluded transactions indicate that users' transactions were received but
    intentionally not included in blocks, often due to fee considerations.

    Args:
        traj_df (pd.DataFrame): Trajectory dataframe with simulation results

    Returns:
        float: Fraction of excluded transactions (0.0 to 1.0)
    """
    return traj_df.iloc[-1].cumm_excl_tx / traj_df.iloc[-1].cumm_total_tx


def base_fee_rebound_inside_range(traj_df: pd.DataFrame,
                                  shock_start: BlocksL1 = -1,
                                  measurement_start: BlocksL1 = -1,
                                  tolerance: Percentage = 0.5) -> bool:
    """
    Determine if the base fee successfully rebounds to within tolerance after a shock.

    This metric evaluates the stability of the fee mechanism by checking if fees
    return to their pre-shock levels after an external price shock.

    Args:
        traj_df (pd.DataFrame): Trajectory dataframe with simulation results
        shock_start (BlocksL1): Block number when shock begins, default is 1/3 of trajectory
        measurement_start (BlocksL1): Block number when to start measuring recovery, default is 2/3 of trajectory
        tolerance (Percentage): Allowable percentage difference for successful rebound

    Returns:
        bool: True if fee rebounded to within tolerance of pre-shock level, False otherwise
    """

    # If shock_start not specified, use the 1/3 point of the trajectory
    if shock_start < 0:
        shock_start = int(traj_df.l1_blocks_passed.quantile(0.33))
    # If measurement_start not specified, use the 2/3 point of the trajectory
    if measurement_start < 0:
        shock_start = int(traj_df.l1_blocks_passed.quantile(0.66))

    # Calculate average base fee before the shock
    avg_base_fee_pre_shock = traj_df.query(
        f'l1_blocks_passed < {shock_start}').base_fee.mean()
    # Calculate average base fee after the measurement point
    avg_base_fee_after_measurement = traj_df.query(
        f'l1_blocks_passed >= {measurement_start}').base_fee.mean()

    # Calculate the relative difference between pre-shock and post-measurement fees
    relative_diff = abs(avg_base_fee_after_measurement -
                        avg_base_fee_pre_shock) / avg_base_fee_pre_shock

    # Return whether the difference is within tolerance
    if relative_diff < tolerance:
        return True
    else:
        return False


def average_base_fee_divided_by_oracle_parameter(traj_df: pd.DataFrame,
                                                 oracle_parameter: str) -> float:
    """
    Calculate the average elasticity of base fee with respect to an oracle parameter.

    This metric measures how responsive the base fee is to changes in a specific
    oracle parameter, which helps evaluate the fee mechanism's adaptiveness.

    Args:
        traj_df (pd.DataFrame): Trajectory dataframe with simulation results
        oracle_parameter (str): Name of the oracle parameter column

    Returns:
        float: Average elasticity value
    """
    return (traj_df['base_fee'].diff() / traj_df[oracle_parameter]).mean()


def counterfactual_sequencer_losses_due_to_lag(traj_df: pd.DataFrame) -> float:
    """
    Calculate the theoretical sequencer losses due to oracle price lag.

    This metric estimates how much sequencers would lose due to the difference
    between market prices and oracle prices, which impacts sequencer economics.

    Args:
        traj_df (pd.DataFrame): Trajectory dataframe with simulation results

    Returns:
        float: Average counterfactual loss in Gwei
    """
    # Calculate the cost in market rates (what sequencers actually pay)
    gwei_per_mana_market = traj_df['base_fee'] / \
        traj_df['market_price_juice_per_gwei']
    # Calculate the cost in oracle rates (what users pay based on)
    gwei_per_mana_oracle = traj_df['base_fee'] / \
        traj_df['oracle_price_juice_per_gwei']
    # The difference represents the loss (or gain) due to lag
    return (gwei_per_mana_market - gwei_per_mana_oracle).mean()


def ratio_of_blocks_with_enough_signatures_per_collected_signatures(traj_df: pd.DataFrame) -> float:
    """
    Calculate the ratio of blocks that received enough signatures to those that
    collected any signatures.

    This metric evaluates the effectiveness of the signature collection mechanism,
    which is important for the consensus security of the system.

    Args:
        traj_df (pd.DataFrame): Trajectory dataframe with simulation results

    Returns:
        float: Ratio of blocks with enough signatures to blocks with any signatures
    """
    # type: ignore
    return float(traj_df.iloc[-1]['cumm_blocks_with_enough_signatures'] / traj_df.iloc[-1]['cumm_blocks_with_collected_signatures'])


def avg_over_fn(group_traj_dfs: list[pd.DataFrame], fn):
    """
    Apply a function to each trajectory and calculate the average result.

    This helper function allows metrics to be aggregated across multiple trajectories.

    Args:
        group_traj_dfs (list[pd.DataFrame]): List of trajectory dataframes
        fn: Function to apply to each trajectory

    Returns:
        float: Average of the function results across all trajectories
    """
    avgs = []
    for traj_df in group_traj_dfs:
        avgs.append(fn(traj_df))
    return np.mean(avgs)


def under_threshold_over_fn(group_traj_dfs: list[pd.DataFrame], fn):
    """
    Calculate the fraction of trajectories where a function's result is below the median.

    This helper function is useful for identifying how many trajectories perform
    better than the median for a particular metric.

    Args:
        group_traj_dfs (list[pd.DataFrame]): List of trajectory dataframes
        fn: Function to apply to each trajectory

    Returns:
        float: Fraction of trajectories where the function result is below the median
    """
    values = []
    for traj_df in group_traj_dfs:
        value = fn(traj_df)
        values.append(value)

    values = np.array(values)
    threshold = np.median(values)
    count_under_threshold = np.sum(values <= threshold)
    return count_under_threshold / len(values)


def elasticity_base_fee_proving_cost(df):
    """
    Calculate the elasticity of base fee with respect to proving cost.

    This metric measures how responsive the base fee is to changes in proving costs,
    which is important for the economic stability of the system.

    Args:
        df (pd.DataFrame): Trajectory dataframe with simulation results

    Returns:
        float: Elasticity of base fee with respect to proving cost
    """
    return average_base_fee_divided_by_oracle_parameter(df, 'PROVING_COST_MODIFICATION_E')


def elasticity_base_fee_fee_juice_price(df):
    """
    Calculate the elasticity of base fee with respect to Juice price.

    This metric measures how responsive the base fee is to changes in the Juice token price,
    which impacts fee stability from a user perspective.

    Args:
        df (pd.DataFrame): Trajectory dataframe with simulation results

    Returns:
        float: Elasticity of base fee with respect to Juice price
    """
    return average_base_fee_divided_by_oracle_parameter(df, 'FEE_JUICE_PRICE_MODIFICATION_E')


def block_average_over_average_mana(traj_df) -> float:
    """
    Calculate the average mana usage per block across finalized epochs.

    This metric measures how efficiently blocks are using their mana capacity,
    which is important for throughput optimization and fee stability.

    Args:
        traj_df (pd.DataFrame): Trajectory dataframe with simulation results

    Returns:
        float: Average mana per block, or NaN if no finalized blocks
    """
    # Find epochs that have been finalized
    finalized_epochs_inds = (traj_df.apply(
        lambda x: x.last_epoch.finalized_time_in_l1 == x.l1_blocks_passed, axis='columns'))
    finalized_epochs = traj_df[finalized_epochs_inds].current_epoch.tolist()

    # Collect mana usage from all blocks in finalized epochs
    # block_averages = []
    block_manas = []
    for epoch in finalized_epochs:
        for block in epoch.slots:
            if block.has_proposal_on_network:
                # block_averages.append(block.tx_total_mana / block.tx_count)
                block_manas.append(block.tx_total_mana)

    # Return the average mana across blocks, or NaN if no blocks
    # if len(block_averages) > 0:
    #     return sum(block_averages) / len(block_averages)
    if len(block_manas) > 0:
        return sum(block_manas) / len(block_manas)
    else:
        return float('nan')


def between_threshold_over_fn(group_traj_dfs: list[pd.DataFrame],
                              fn,
                              threshold_col: str,
                              lower_threshold: float = 0.9,
                              upper_threshold: float = 1.1,
                              ) -> float:
    """
    Calculate the fraction of trajectories where a function's result is between thresholds.

    This helper function identifies how many trajectories have a metric value 
    within a specified range relative to a reference column.

    Args:
        group_traj_dfs (list[pd.DataFrame]): List of trajectory dataframes
        fn: Function to apply to each trajectory
        threshold_col (str): Column name to use as reference for thresholds
        lower_threshold (float): Lower bound multiplier for the reference column
        upper_threshold (float): Upper bound multiplier for the reference column

    Returns:
        float: Fraction of trajectories where the function result is between thresholds
    """
    success_count = 0
    total_count = 0
    for traj_df in group_traj_dfs:
        total_count = 0
        value = fn(traj_df)
        lower_bound = traj_df[threshold_col] * lower_threshold
        upper_bound = traj_df[threshold_col] * upper_bound
        if (value >= lower_bound) & (value < upper_bound):
            success_count += 1
    return success_count / total_count


def tg_m12(group_traj_dfs: list[pd.DataFrame],
           lower_tol: float = 0.665,
           upper_tol: float = 1.334) -> float:
    """
    Calculate the fraction of trajectories where average mana usage is within
    target range.

    This metric evaluates how well the system is maintaining mana usage near
    the target level, which is important for fee stability and throughput.

    Args:
        group_traj_dfs (list[pd.DataFrame]): List of trajectory dataframes
        lower_tol (float): Lower tolerance factor relative to target mana
        upper_tol (float): Upper tolerance factor relative to target mana

    Returns:
        float: Fraction of trajectories with average mana within target range
    """
    success_count = 0
    total_count = 0

    for traj_df in group_traj_dfs:
        total_count += 1
        # Calculate average mana per block
        value = block_average_over_average_mana(traj_df)
        # Calculate target mana from model parameters
        target_mana = (traj_df['RELATIVE_TARGET_MANA_PER_BLOCK']
                       * traj_df['MAXIMUM_MANA_PER_BLOCK']).iloc[-1]
        # Calculate bounds based on tolerance
        lower_bound = target_mana * lower_tol
        upper_bound = target_mana * upper_tol

        # Check if average mana is within bounds
        if (value >= lower_bound) & (value < upper_bound):
            success_count += 1

    return success_count / total_count


def tg_m13(group_traj_dfs: list[pd.DataFrame],
           lower_tol: float = 0.2,
           upper_tol: float = 1.0) -> float:
    """
    Calculate the fraction of trajectories where average mana usage is within
    maximum capacity range.

    This metric evaluates how efficiently the system is utilizing the maximum
    available mana capacity, balancing between underuse and saturation.

    Args:
        group_traj_dfs (list[pd.DataFrame]): List of trajectory dataframes
        lower_tol (float): Lower tolerance factor relative to maximum mana
        upper_tol (float): Upper tolerance factor relative to maximum mana

    Returns:
        float: Fraction of trajectories with average mana within capacity range
    """
    success_count = 0
    total_count = 0

    for traj_df in group_traj_dfs:
        total_count += 1
        # Calculate average mana per block
        value = block_average_over_average_mana(traj_df)
        # Get maximum mana capacity
        max_mana = traj_df['MAXIMUM_MANA_PER_BLOCK'].iloc[-1]
        # Calculate bounds based on tolerance
        lower_bound = max_mana * lower_tol
        upper_bound = max_mana * upper_tol

        # Check if average mana is within bounds
        if (value >= lower_bound) & (value < upper_bound):
            success_count += 1

    return success_count / total_count


# Dictionary mapping individual trajectory metric codes to descriptive labels
PER_TRAJECTORY_METRICS_LABELS = {
    'T-M1': "Fee/Juice Volatility",
    'T-M2': "Empty Blocks",
    'T-M3': "Unproven Epochs",
    'T-M4': "Percentage of Dropped Transactions during Trajectory",
    'T-M5': "Percentage of Excluded Transactions during Trajectory",
    'T-M6': "Base Fee Rebound is inside range",
    'T-M7a': "Average Elasticity of Base Fee by Proving Cost",
    'T-M7b': "Average Elasticity of Base Fee by Fee Juice Price",
    'T-M8': "Counterfactual Sequencer Losses due to Lag",
    'T-M9': "Ratio of Blocks with Enough Signatures per Blocks with Signatures",
    'T-M10': "Block-Average of Average Mana used per Block on the last timestep",
}

# Dictionary mapping group trajectory metric codes to descriptive labels
PER_TRAJECTORY_GROUP_METRICS_LABELS = {
    'TG-M1': "Trajectory-Average over Relative Volatility",
    'TG-M2': "Trajectory-Average over Empty Blocks",
    'TG-M3': "Trajectory-Average over Unproven Epochs Across",
    'TG-M4': "Trajectory-Average over percentage of dropped transactions",
    'TG-M5': "Percentage of MC runs above dropped threshold",
    'TG-M6': "Trajectory-Average over percentage of excluded transactions",
    'TG-M7': "Trajectory-Average over Rebound being sucessful",
    'TG-M8a': "Trajectory-Average over Elasticity of Base Fee with respect to Proving Cost",
    'TG-M8b': "Trajectory-Average over Elasticity of Base Fee with respect to Fee Juice Price",
    'TG-M9': "Trajectory-Average over Counterfactual Sequencer Losses",
    'TG-M10': "Trajectory-Average over Ratio of Blocks with Enough Signatures per Blocks with Signatures",
    'TG-M12': "Percentage of Trajectories where Block-Average Mana used is within range of target mana",
    'TG-M13': "Percentage of Trajectories where Block-Average Mana used is within range of max mana",
}

# Dictionary mapping individual trajectory metric codes to their functions
PER_TRAJECTORY_METRICS = {
    'T-M1': relative_volatility_across_time,
    'T-M2': empty_blocks_during_trajectory,
    'T-M3': unproven_epochs_during_trajectory,
    'T-M4': fraction_dropped_tx_during_trajectory,
    'T-M5': fraction_excluded_tx_during_trajectory,
    'T-M6': base_fee_rebound_inside_range,
    'T-M7a': elasticity_base_fee_proving_cost,
    'T-M7b': elasticity_base_fee_fee_juice_price,
    'T-M8': counterfactual_sequencer_losses_due_to_lag,
    'T-M9': ratio_of_blocks_with_enough_signatures_per_collected_signatures,
    'T-M10': block_average_over_average_mana
}

# Dictionary mapping group trajectory metric codes to their aggregation functions
PER_TRAJECTORY_GROUP_METRICS = {
    'TG-M1': lambda dfs: avg_over_fn(dfs, relative_volatility_across_time),
    'TG-M2': lambda dfs: avg_over_fn(dfs, empty_blocks_during_trajectory),
    'TG-M3': lambda dfs: avg_over_fn(dfs, unproven_epochs_during_trajectory),
    'TG-M4': lambda dfs: avg_over_fn(dfs, fraction_dropped_tx_during_trajectory),
    'TG-M5': lambda dfs: under_threshold_over_fn(dfs, fraction_dropped_tx_during_trajectory),
    'TG-M6': lambda dfs: avg_over_fn(dfs, fraction_excluded_tx_during_trajectory),
    'TG-M7': lambda dfs: avg_over_fn(dfs, base_fee_rebound_inside_range),
    'TG-M8a': lambda dfs: avg_over_fn(dfs, elasticity_base_fee_proving_cost),
    'TG-M8b': lambda dfs: avg_over_fn(dfs, elasticity_base_fee_fee_juice_price),
    'TG-M9': lambda dfs: avg_over_fn(dfs, counterfactual_sequencer_losses_due_to_lag),
    'TG-M10': lambda dfs: avg_over_fn(dfs, ratio_of_blocks_with_enough_signatures_per_collected_signatures),
    'TG-M12': tg_m12,  
    'TG-M13': tg_m13, 
}

# Dictionary mapping group metrics to boolean comparison functions for aggregation
PER_TRAJECTORY_GROUP_COLLAPSED_METRICS = {
    'TG-M1': lambda agg_df, x: agg_df[x] < agg_df[x].median(),
    'TG-M2': lambda agg_df, x: agg_df[x] < agg_df[x].median(),
    'TG-M3': lambda agg_df, x: agg_df[x] < agg_df[x].median(),
    'TG-M4': lambda agg_df, x: agg_df[x] < agg_df[x].median(),
    'TG-M5': lambda agg_df, x: agg_df[x] > agg_df[x].median(),
    'TG-M6': lambda agg_df, x: agg_df[x] < agg_df[x].median(),
    'TG-M7': lambda agg_df, x: agg_df[x] > agg_df[x].median(),
    'TG-M8a': lambda agg_df, x: agg_df[x] > agg_df[x].median(),
    'TG-M8b': lambda agg_df, x: agg_df[x] > agg_df[x].median(),
    'TG-M9': lambda agg_df, x: agg_df[x] < agg_df[x].median(),
    'TG-M10': lambda agg_df, x: agg_df[x] > agg_df[x].median(),
    'TG-M12': lambda agg_df, x: agg_df[x] > agg_df[x].median(),
    'TG-M13': lambda agg_df, x: agg_df[x] > agg_df[x].median(),
}


def retrieve_feature_df(sim_df, control_params, RELEVANT_PER_TRAJECTORY_GROUP_METRICS):
    """
    Extract feature dataframes from simulation results for specified metrics.

    This function processes simulation results to create dataframes of metric values
    that can be used for further analysis and visualization.

    Args:
        sim_df (pd.DataFrame): DataFrame with simulation results
        control_params (list): List of parameter names used as control variables
        RELEVANT_PER_TRAJECTORY_GROUP_METRICS (list): List of metric codes to include

    Returns:
        tuple: (agg_df, collapsed_agg_df) - Raw and binary-collapsed metric dataframes
    """
    # Define grouping parameters for aggregation
    group_params = ['simulation', 'subset'] + control_params
    records = []

    # Calculate each relevant metric for each group
    for label in RELEVANT_PER_TRAJECTORY_GROUP_METRICS:
        fn = PER_TRAJECTORY_GROUP_METRICS[label]

        # Group trajectories by control parameters
        groups = list(sim_df.reset_index().groupby(group_params))
        for i, g in groups:
            # Get list of dataframes for each run in the group
            dfs = [el[1] for el in list(g.groupby('run'))]
            # Apply the metric function to the group
            value = fn(dfs)
            # Create a record with group parameters and metric value
            record = dict(zip(group_params, i))
            record['metric'] = label
            record['metric_value'] = value
            records.append(record)

    # Create a dataframe with metrics in columns
    agg_df = pd.DataFrame(records).groupby(
        group_params + ['metric']).metric_value.first().unstack().reset_index()

    # Apply binary collapsing to the metrics
    collapsed_agg_df = compute_agg_df(
        RELEVANT_PER_TRAJECTORY_GROUP_METRICS, agg_df)
    return agg_df, collapsed_agg_df


def compute_agg_df(RELEVANT_PER_TRAJECTORY_GROUP_METRICS, agg_df):
    """
    Convert raw metric values to binary indicators based on median comparisons.

    This function transforms continuous metric values into binary indicators
    (better/worse than median) for simplified analysis and visualization.

    Args:
        RELEVANT_PER_TRAJECTORY_GROUP_METRICS (list): List of metric codes to process
        agg_df (pd.DataFrame): DataFrame with raw metric values

    Returns:
        pd.DataFrame: DataFrame with binary (True/False) metric indicators
    """
    collapsed_agg_df = agg_df.copy()
    for label in RELEVANT_PER_TRAJECTORY_GROUP_METRICS:
        # Apply the appropriate comparison function to each metric
        collapsed_agg_df[label] = PER_TRAJECTORY_GROUP_COLLAPSED_METRICS[label](
            agg_df, label)
    return collapsed_agg_df
