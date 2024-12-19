from aztec_gddt.logic import *
from copy import deepcopy

MODEL_BLOCKS = [
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
        'label': 'Evolve Slot State',
        'ignore': False,
        'policies': {

        },
        'variables': {

        }
    },
    {
        'label': 'Current',
        'ignore': False,
        'policies': {

        },
        'variables': {

        }
    }
]


def p_epoch(params: ModelParams, _2, _3, state: ModelState):
    """
    Logic for the evolution over the epoch/slot state
    """
    last_epoch = state['last_epoch']
    epoch = state['current_epoch']
    curr_slot = epoch.slots[-1]

    l1_blocks_since_slot_init = state['l1_blocks_passed'] - \
        curr_slot.init_time_in_l1

    if l1_blocks_since_slot_init < params['general'].L1_SLOTS_PER_L2_SLOT:
        if l1_blocks_since_slot_init >= curr_slot.time_until_E_BLOCK_SENT:
            curr_slot.has_block_header_on_l1 = True

        if l1_blocks_since_slot_init >= curr_slot.time_until_E_BLOCK_VALIDATE:
            curr_slot.has_validator_signatures = True

        if l1_blocks_since_slot_init >= curr_slot.time_until_E_BLOCK_PROPOSE:
            curr_slot.has_proposal_on_network = True
    else:
        # Move on to the next slot or epoch
        t1 = 0.75  # TODO
        t2 = 0.25  # TODO
        t3 = 0.5  # TODO

        i_slot = len(epoch.slots)
        if len(epoch.slots) < params['general'].L2_SLOTS_PER_L2_EPOCH:
            proposer = epoch.proposers[i_slot]

            # NOTE: slot is created here
            new_slot = Slot(state['l1_blocks_passed'],
                            proposer,
                            t1,
                            t1 + t2,
                            t1 + t2 + t3)
            epoch.slots.append(new_slot)
        else:

            last_epoch = deepcopy(epoch)

            # For each slot in the epoch a sequencer/block proposer is drawn (based on score) from the validator committee
            new_proposers = [str(i) for i in range(0, 50)]  # TODO


            # 300 validators are drawn (based on score) to the validator committee from the validator set (i.e. from the set of staked users)
            new_validators = [str(i) for i in range(0, 50)]  # TODO
            proposer = new_proposers[0]

            # NOTE: slot is created here
            new_slot = Slot(state['l1_blocks_passed'],
                            proposer,
                            t1,
                            t1 + t2,
                            t1 + t2 + t3)

            t4 = 5
            t5 = t4 + 1

            # NOTE: epoch is created here
            epoch = Epoch(state['l1_blocks_passed'],
                          new_proposers,
                          new_validators,
                          new_slot,
                          [],
                          t4,
                          t5)

    return {'current_epoch': epoch,
            'last_epoch': last_epoch}

    # Check if all blocks are done

    if curr_slot.is_valid_proposal:
        epoch.slots.append(new_slot)
        # TODO: create new slot

    # 1. Update slot state based on BlockEvolutionState
    # 2. If time is over or slot is pending-finish, create new block


MODEL_BLOCKS = [block for block in MODEL_BLOCKS
                if block.get('ignore', False) is False]
