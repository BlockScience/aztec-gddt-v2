from aztec_gddt.types import *

def p_evolve_time(params: ModelParams, _2, _3, _4):
    return {'delta_days': params['timestep_in_l1_blocks']}

def s_days_passed(_1, _2, _3,
                  state: ModelState,
                  signal):
    return ('days_passed', state['l1_blocks_passed'] + signal['delta_l1_blocks'])

def s_delta_days(_1, _2, _3, _4, signal):
    return ('delta_days', signal['delta_l1_blocks'])








def block_reward(time_in_l1_blocks: BlocksL1,
                last_reward_time_in_l1_blocks: BlocksL1,
                last_reward: Token) -> Token:
    return 0.0


def transaction_fee(fee_params: FeeParams) -> Wei:

    # L1_gas_per_L2_block: Gas = L1_gas_per_block_proposed + blobs_per_block * POINT_EVALUATION_PRECOMPILE_GAS
    # wei_per_L2_block: Wei = L1_gas_per_L2_block * wei_per_L1_gas

    # wei_for_DA_per_L2_block: Wei = blobs_per_block * L1_gas_per_blob * wei_per_L1_blob_gas


    # L1_cost_per_mana = L1_cost_per_L2_cost / fee_params.TARGET_MANA_PER_BLOCK

    return 0