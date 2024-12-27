import pandas as pd
from aztec_gddt.types import *

def relative_volatility_across_time(traj_df: pd.DataFrame) -> float:
    return traj_df.market_price_juice_per_mana.std() / traj_df.base_fee.std()

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

    avg_base_fee_pre_shock = traj_df.query(f'l1_blocks_passed < {shock_start}').base_fee.mean()
    avg_base_fee_after_measurement = traj_df.query(f'l1_blocks_passed >= {measurement_start}').base_fee.mean()


    relative_diff = abs(avg_base_fee_after_measurement - avg_base_fee_pre_shock) / avg_base_fee_pre_shock

    if relative_diff < tolerance:
        return True
    else:
        return False

def base_fee_divided_by_oracle_parameter(traj_df: pd.DataFrame) -> float:
    return float('nan')

def counterfactual_sequencer_losses_due_to_lag(traj_df: pd.DataFrame) -> float:
    return float('nan')

def network_resumed_finalization_following_inactivity(traj_df: pd.DataFrame) -> bool:
    return False


PER_TRAJECTORY_METRICS = {
    'T-M1': relative_volatility_across_time,
    'T-M2': empty_blocks_during_trajectory,
    'T-M3': unproven_epochs_during_trajectory,
    'T-M4': fraction_dropped_tx_during_trajectory,
    'T-M5': fraction_excluded_tx_during_trajectory,
    'T-M6': base_fee_rebound_inside_range,
    'T-M7': base_fee_divided_by_oracle_parameter,
    'T-M8': counterfactual_sequencer_losses_due_to_lag,
    'T-M9': network_resumed_finalization_following_inactivity,
}

PER_TRAJECTORY_GROUP_METRICS = {

}