"""
Post-processing utilities for Aztec GDDT simulation results.

This module provides functions for calculating derived metrics from simulation results.
These post-processing operations transform raw simulation data into more useful metrics for analysis and evaluation of the Aztec rollup system's economic properties.

Key functionalities:
1. Value at risk calculations for validator stakes
2. Normalization of congestion multipliers
3. Average mana usage calculations relative to targets
"""

import pandas as pd
import numpy as np


def value_at_risk_in_usd(row, q):
    """
    Calculate the Value at Risk (VaR) in USD for the validator stakes.

    This function computes the total value of the validator stakes. It represents the value at risk for the current epoch.

    Args:
        row: DataFrame row containing simulation state
        q (float): Quantile threshold (0.0 to 1.0) for stake calculation

    Returns:
        float: Value at risk in USD, or NaN if no active stakes or price is zero
    """
    # Extract and sort all validator stakes from smallest to largest
    active_stakes_in_epoch = np.array(
        sorted(row['agents'][a].stake for a in row.current_epoch.validators))

    # Only proceed if there are active stakes and the price conversion is possible
    if len(active_stakes_in_epoch) > 0 and (row['market_price_juice_per_gwei'] > 0):
        value_at_risk_in_juice = active_stakes_in_epoch[active_stakes_in_epoch <= np.quantile(
            active_stakes_in_epoch, q)].sum()
        value_at_risk_in_gwei = value_at_risk_in_juice / \
            row['market_price_juice_per_gwei']
        value_at_risk_in_eth = value_at_risk_in_gwei / 1e9

        # Convert from ETH to USD using market price
        value_at_risk_in_usd = value_at_risk_in_eth * row['market_price_eth']
    else:
        # Return NaN if calculation isn't possible
        value_at_risk_in_usd = float('nan')

    return value_at_risk_in_usd


def post_process_sim_df(sim_df) -> pd.DataFrame:
    """
    Add derived metrics to the simulation results dataframe.

    This function calculates various derived metrics that are useful for 
    analyzing the simulation results, such as normalized congestion multipliers, 
    mana usage ratios, and value at risk measures.

    Args:
        sim_df (pd.DataFrame): Raw simulation results dataframe

    Returns:
        pd.DataFrame: Enhanced dataframe with additional derived metrics
    """
    # Calculate normalized congestion multiplier (relative to minimum)
    sim_df['normed_congestion_multiplier'] = sim_df['congestion_multiplier'] / \
        sim_df['MINIMUM_MULTIPLIER_CONGESTION']

    # Calculate average mana per block as a ratio of target mana
    # This measures how close the system is to its target utilization
    sim_df['average_mana_per_block_per_target'] = sim_df.apply(lambda df: sum(b.tx_total_mana for b in df.last_epoch.slots) / len(df.last_epoch.slots) / (
        df.MAXIMUM_MANA_PER_BLOCK * df.RELATIVE_TARGET_MANA_PER_BLOCK) if len(df.last_epoch.slots) > 0 else float('nan'), axis='columns')

    # Calculate average mana per block as a ratio of maximum capacity
    # This measures how efficiently the system is utilizing its maximum capacity
    sim_df['average_mana_per_block_per_max'] = sim_df.apply(lambda df: sum(b.tx_total_mana for b in df.last_epoch.slots) / len(
        df.last_epoch.slots) / (df.MAXIMUM_MANA_PER_BLOCK) if (len(df.last_epoch.slots) > 0 & df.last_epoch.finalized) else float('nan'), axis='columns')
    # Calculate Value at Risk metrics at different required signature percentages.
    sim_df['value_at_risk_in_usd_q33'] = sim_df.apply(
        lambda row: value_at_risk_in_usd(row, q=0.33), axis='columns')
    sim_df['value_at_risk_in_usd_q50'] = sim_df.apply(
        lambda row: value_at_risk_in_usd(row, q=0.50), axis='columns')
    sim_df['value_at_risk_in_usd_q66'] = sim_df.apply(
        lambda row: value_at_risk_in_usd(row, q=0.66), axis='columns')
    return sim_df
