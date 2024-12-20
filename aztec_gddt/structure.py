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
            'days_passed': s_days_passed,
            'delta_days': s_delta_days
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
            'last_epoch': replace_suf

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