from aztec_gddt.logic import *
from copy import deepcopy

RAW_MODEL_BLOCKS: list[dict] = [
    {
        'label': 'Time Tracking',
        'ignore': False,
        'desc': 'Updates the time in the system',
        'policies': {
            'evolve_time': p_evolve_time
        },
        'variables': {
            'l1_blocks_passed': s_blocks_passed,
            'delta_l1_blocks': s_delta_blocks
        }
    },
    {
        'label': 'Set Exogenous Variables',
        'policies': {

        },
        'variables': {
            'market_price_juice_per_mana': s_market_price_juice_per_mana,
            'market_price_l1_gas': s_market_price_l1_gas,
            'market_price_l1_blobgas': s_market_price_l1_blobgas,
        }
    },
    {
        'label': 'Oracles & Scoring Functions',
        'ignore': False,
        'policies': {
            'juice_per_mana': p_oracle_juice_per_mana,
            'l1_gas': p_oracle_l1_gas,
            'l1_blobgas': p_oracle_l1_blobgas,
            'proving_cost': p_oracle_proving_cost

        },
        'variables': {
            'oracle_price_juice_per_mana': replace_suf,
            'oracle_price_l1_gas': replace_suf,
            'oracle_price_l1_blobgas': replace_suf,
            'update_time_oracle_price_juice_per_mana': replace_suf,
            'update_time_oracle_price_l1_gas': replace_suf,
            'update_time_oracle_price_l1_blobgas': replace_suf,
            'oracle_proving_cost': replace_suf

        }
    },
    {
        'label': 'Contract Values',
        'ignore': False,
        'policies': {

        },
        'variables': {
            'congestion_multiplier': s_congestion_multiplier
        }
    },
    {
        'label': 'Epoch/Slot evolution',
        'ignore': False,
        'policies': {
            'evolve_epoch_slot': p_epoch

        },
        'variables': {
            'current_epoch': replace_suf,
            'last_epoch': replace_suf,
            'cumm_dropped_tx': add_suf,
            'cumm_excl_tx': add_suf,
            'cumm_total_tx': add_suf,
            'excess_mana': replace_suf,
            'l2_blocks_passed': add_suf,
            'base_fee': replace_suf
        }
    },
    {
        'label': 'Epoch Proving',
        'ignore': False,
        'policies': {
            'epoch_proving': p_pending_epoch_proof
        },
        'variables': {
            'last_epoch': replace_suf,
            'last_reward_time_in_l1': replace_suf,
            'last_reward': replace_suf,
            'cumm_empty_blocks': add_suf,
            'cumm_unproven_epochs': add_suf,
            'cumm_resolved_epochs': add_suf,
            'cumm_finalized_epochs': add_suf,
            'cumm_mana_used_on_finalized_blocks': add_suf,
            'cumm_finalized_blocks': add_suf,
            'agents': replace_suf
        }
    }
]


blocks: list[dict] = []
for block in [b for b in RAW_MODEL_BLOCKS if b.get("ignore", False) != True]:
    _block: dict = deepcopy(block)
    for variable, suf in block.get("variables", {}).items():  # type: ignore
        if suf == add_suf:
            _block["variables"][variable] = add_suf(variable)  # type: ignore
        elif suf == replace_suf:
            _block["variables"][variable] = replace_suf(
                variable)  # type: ignore

    blocks.append(_block)


MODEL_BLOCKS = [block for block in blocks
                if block.get('ignore', False) is False]
