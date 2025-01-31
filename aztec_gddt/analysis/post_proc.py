import pandas as pd

def post_process_sim_df(sim_df) -> pd.DataFrame:
    # Post Processing Metrics
    sim_df['normed_congestion_multiplier'] = sim_df['congestion_multiplier'] / \
        sim_df['MINIMUM_MULTIPLIER_CONGESTION']
    sim_df['average_mana_per_block_per_target'] = sim_df.apply(lambda df: sum(b.tx_total_mana for b in df.last_epoch.slots) / len(df.last_epoch.slots) / (
        df.MAXIMUM_MANA_PER_BLOCK * df.RELATIVE_TARGET_MANA_PER_BLOCK) if len(df.last_epoch.slots) > 0 else float('nan'), axis='columns')
    sim_df['average_mana_per_block_per_max'] = sim_df.apply(lambda df: sum(b.tx_total_mana for b in df.last_epoch.slots) / len(
        df.last_epoch.slots) / (df.MAXIMUM_MANA_PER_BLOCK) if (len(df.last_epoch.slots) > 0 & df.last_epoch.finalized) else float('nan'), axis='columns')
    return sim_df