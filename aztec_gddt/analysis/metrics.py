import pandas as pd
import numpy as np
from aztec_gddt.types import *


def relative_volatility_across_time(traj_df: pd.DataFrame) -> float:
    return traj_df.market_price_juice_per_gwei.std() / traj_df.base_fee.std()


def empty_blocks_during_trajectory(traj_df: pd.DataFrame) -> int:
    return traj_df.iloc[-1].cumm_empty_blocks


def unproven_epochs_during_trajectory(traj_df: pd.DataFrame) -> int:
    return traj_df.iloc[-1].cumm_unproven_epochs


def fraction_dropped_tx_during_trajectory(traj_df: pd.DataFrame) -> float:
    return traj_df.iloc[-1].cumm_dropped_tx / traj_df.iloc[-1].cumm_total_tx


def fraction_excluded_tx_during_trajectory(traj_df: pd.DataFrame) -> float:
    return traj_df.iloc[-1].cumm_excl_tx / traj_df.iloc[-1].cumm_total_tx


def base_fee_rebound_inside_range(traj_df: pd.DataFrame,
                                  shock_start: BlocksL1 = -1,
                                  measurement_start: BlocksL1 = -1,
                                  tolerance: Percentage = 0.5) -> bool:
    # TODO check assumptions on default values
    # TODO make sure that they're properly parametrized across the workflow

    if shock_start < 0:
        shock_start = int(traj_df.l1_blocks_passed.quantile(0.33))
    if measurement_start < 0:
        shock_start = int(traj_df.l1_blocks_passed.quantile(0.66))

    avg_base_fee_pre_shock = traj_df.query(
        f'l1_blocks_passed < {shock_start}').base_fee.mean()
    avg_base_fee_after_measurement = traj_df.query(
        f'l1_blocks_passed >= {measurement_start}').base_fee.mean()

    relative_diff = abs(avg_base_fee_after_measurement -
                        avg_base_fee_pre_shock) / avg_base_fee_pre_shock

    if relative_diff < tolerance:
        return True
    else:
        return False


def average_base_fee_divided_by_oracle_parameter(traj_df: pd.DataFrame,
                                                 oracle_parameter: str) -> float:
    return (traj_df['base_fee'].diff() / traj_df[oracle_parameter]).mean()


def counterfactual_sequencer_losses_due_to_lag(traj_df: pd.DataFrame) -> float:
    # XXX
    gwei_per_mana_market = traj_df['base_fee'] / \
        traj_df['market_price_juice_per_gwei']
    gwei_per_mana_oracle = traj_df['base_fee'] / \
        traj_df['oracle_price_juice_per_gwei']
    return (gwei_per_mana_market - gwei_per_mana_oracle).mean()


def ratio_of_blocks_with_enough_signatures_per_collected_signatures(traj_df: pd.DataFrame) -> float:
    # type: ignore
    return float(traj_df.iloc[-1]['cumm_blocks_with_enough_signatures'] / traj_df.iloc[-1]['cumm_blocks_with_collected_signatures'])


def avg_over_fn(group_traj_dfs: list[pd.DataFrame], fn):
    avgs = []
    for traj_df in group_traj_dfs:
        avgs.append(fn(traj_df))
    return np.mean(avgs)


def under_threshold_over_fn(group_traj_dfs: list[pd.DataFrame], fn):
    values = []
    for traj_df in group_traj_dfs:
        value = fn(traj_df)
        values.append(value)

    values = np.array(values)
    threshold = np.median(values)
    count_under_threshold = np.sum(values <= threshold)
    return count_under_threshold / len(values)


def elasticity_base_fee_proving_cost(
    df): return average_base_fee_divided_by_oracle_parameter(df, 'PROVING_COST_MODIFICATION_E')


def elasticity_base_fee_fee_juice_price(
    df): return average_base_fee_divided_by_oracle_parameter(df, 'FEE_JUICE_PRICE_MODIFICATION_E')


def block_average_over_average_mana(traj_df) -> float:
    """

    """
    finalized_epochs_inds = (traj_df.apply(lambda x: x.last_epoch.finalized_time_in_l1 == x.l1_blocks_passed, axis='columns'))
    finalized_epochs = traj_df[finalized_epochs_inds].current_epoch.tolist()

    # block_averages = []
    block_manas = []
    for epoch in finalized_epochs:
        for block in epoch.slots:
            if block.has_proposal_on_network:
                # block_averages.append(block.tx_total_mana / block.tx_count)
                block_manas.append(block.tx_total_mana)
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
    success_count = 0
    total_count = 0

    for traj_df in group_traj_dfs:
        total_count += 1
        value = block_average_over_average_mana(traj_df)
        target_mana = (traj_df['RELATIVE_TARGET_MANA_PER_BLOCK']
                       * traj_df['MAXIMUM_MANA_PER_BLOCK']).iloc[-1]
        lower_bound = target_mana * lower_tol
        upper_bound = target_mana * upper_tol

        if (value >= lower_bound) & (value < upper_bound):
            success_count += 1

    return success_count / total_count


def tg_m13(group_traj_dfs: list[pd.DataFrame],
           lower_tol: float = 0.2,
           upper_tol: float = 1.0) -> float:
    success_count = 0
    total_count = 0

    for traj_df in group_traj_dfs:
        total_count += 1
        value = block_average_over_average_mana(traj_df)
        max_mana = traj_df['MAXIMUM_MANA_PER_BLOCK'].iloc[-1]
        lower_bound = max_mana * lower_tol
        upper_bound = max_mana * upper_tol

        if (value >= lower_bound) & (value < upper_bound):
            success_count += 1

    return success_count / total_count


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

PER_TRAJECTORY_GROUP_METRICS_LABELS = {
    'TG-M1': "Trajectory-Average over Relative Volatility",
    'TG-M2': "Trajectory-Average over Empty Blocks",
    'TG-M3': "Trajectory- over Unproven Epochs Across",
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
    'TG-M12': tg_m12,  # TODO
    'TG-M13': tg_m13,  # TODO
}

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

    group_params = ['simulation', 'subset'] + control_params
    records = []
    for label in RELEVANT_PER_TRAJECTORY_GROUP_METRICS:
        fn = PER_TRAJECTORY_GROUP_METRICS[label]

        groups = list(sim_df.reset_index().groupby(group_params))
        for i, g in groups:
            dfs = [el[1] for el in list(g.groupby('run'))]
            value = fn(dfs)
            record = dict(zip(group_params, i))
            record['metric'] = label
            record['metric_value'] = value
            records.append(record)

    agg_df = pd.DataFrame(records).groupby(
        group_params + ['metric']).metric_value.first().unstack().reset_index()

    collapsed_agg_df = agg_df.copy()

    for label in RELEVANT_PER_TRAJECTORY_GROUP_METRICS:
        collapsed_agg_df[label] = PER_TRAJECTORY_GROUP_COLLAPSED_METRICS[label](
            agg_df, label)
    return agg_df, collapsed_agg_df
