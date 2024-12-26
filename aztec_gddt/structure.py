from aztec_gddt.logic import *
from copy import deepcopy

RAW_MODEL_BLOCKS = [
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
            'cumm_finalized_blocks': add_suf
        }
    }
]





blocks = []
for block in [b for b in RAW_MODEL_BLOCKS if b.get("ignore", False) != True]:
    _block = deepcopy(block)
    for variable, suf in block.get("variables", {}).items(): # type: ignore
        if suf == add_suf:
            _block["variables"][variable] = add_suf(variable) # type: ignore
        elif suf == replace_suf:
            _block["variables"][variable] = replace_suf(variable) # type: ignore

    blocks.append(_block)


MODEL_BLOCKS = [block for block in blocks
                if block.get('ignore', False) is False]