import pandas as pd





def relative_volatility_across_time(traj_df: pd.DataFrame) -> float:
    return traj_df.market_price_juice_per_mana.std() / traj_df.base_fee.std()

def empty_blocks_during_trajectory(traj_df: pd.DataFrame) -> int:
    return traj_df.iloc[-1].cumm_empty_blocks

def unproven_epochs_during_trajectory(traj_df: pd.DataFrame) -> int:
    return traj_df.iloc[-1].cumm_unproven_epochs



PER_TRAJECTORY_METRICS = {
    'T-M1': relative_volatility_across_time,
    'T-M2': empty_blocks_during_trajectory,
    'T-M3': unproven_epochs_during_trajectory
}

PER_TRAJECTORY_GROUP_METRICS = {
    
}