import pandas as pd
import numpy as np

def value_at_risk_in_usd(row, q):
    active_stakes_in_epoch = np.array(sorted(row['agents'][a].stake for a in row.current_epoch.validators))

    if len(active_stakes_in_epoch) > 0 and (row['market_price_juice_per_gwei'] > 0):
        value_at_risk_in_juice = active_stakes_in_epoch[active_stakes_in_epoch <= np.quantile(active_stakes_in_epoch, q)].sum()
        value_at_risk_in_gwei = value_at_risk_in_juice / row['market_price_juice_per_gwei']
        value_at_risk_in_eth = value_at_risk_in_gwei / 1e9
        value_at_risk_in_usd = value_at_risk_in_eth * row['market_price_eth']
    else:
        value_at_risk_in_usd = float('nan')

    return value_at_risk_in_usd

def post_process_sim_df(sim_df) -> pd.DataFrame:
    # Post Processing Metrics
    sim_df['normed_congestion_multiplier'] = sim_df['congestion_multiplier'] / \
        sim_df['MINIMUM_MULTIPLIER_CONGESTION']
    sim_df['average_mana_per_block_per_target'] = sim_df.apply(lambda df: sum(b.tx_total_mana for b in df.last_epoch.slots) / len(df.last_epoch.slots) / (
        df.MAXIMUM_MANA_PER_BLOCK * df.RELATIVE_TARGET_MANA_PER_BLOCK) if len(df.last_epoch.slots) > 0 else float('nan'), axis='columns')
    sim_df['average_mana_per_block_per_max'] = sim_df.apply(lambda df: sum(b.tx_total_mana for b in df.last_epoch.slots) / len(
        df.last_epoch.slots) / (df.MAXIMUM_MANA_PER_BLOCK) if (len(df.last_epoch.slots) > 0 & df.last_epoch.finalized) else float('nan'), axis='columns')
    sim_df['value_at_risk_in_usd_q33'] = sim_df.apply(lambda row: value_at_risk_in_usd(row, q=0.33), axis='columns')
    sim_df['value_at_risk_in_usd_q50'] = sim_df.apply(lambda row: value_at_risk_in_usd(row, q=0.50), axis='columns')
    sim_df['value_at_risk_in_usd_q66'] = sim_df.apply(lambda row: value_at_risk_in_usd(row, q=0.66), axis='columns')
    return sim_df