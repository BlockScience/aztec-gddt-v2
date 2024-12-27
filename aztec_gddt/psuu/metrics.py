import pandas as pd





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


PER_TRAJECTORY_METRICS = {
    'T-M1': relative_volatility_across_time,
    'T-M2': empty_blocks_during_trajectory,
    'T-M3': unproven_epochs_during_trajectory,
    'T-M4': fraction_dropped_tx_during_trajectory,
    'T-M5': fraction_excluded_tx_during_trajectory,
    # 'T-M6': None,
    # 'T-M7': None,
    # 'T-M8': None,
    # 'T-M9': None,
}

PER_TRAJECTORY_GROUP_METRICS = {

}