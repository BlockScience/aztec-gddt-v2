from aztec_gddt.types import *
from copy import deepcopy

def p_evolve_time(params: ModelParams, _2, _3, _4):
    return {'delta_days': params['timestep_in_l1_blocks']}

def s_days_passed(_1, _2, _3,
                  state: ModelState,
                  signal):
    return ('days_passed', state['l1_blocks_passed'] + signal['delta_l1_blocks'])

def s_delta_days(_1, _2, _3, _4, signal):
    return ('delta_days', signal['delta_l1_blocks'])


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


            # For each slot in the epoch a sequencer/block proposer is drawn (based on score) from the validator committee
            proposer = epoch.validators[i_slot]

            # NOTE: slot is created here
            new_slot = Slot(state['l1_blocks_passed'],
                            proposer,
                            t1,
                            t1 + t2,
                            t1 + t2 + t3)
            epoch.slots.append(new_slot)
        else:

            last_epoch = deepcopy(epoch)

            # N validators are drawn (based on score) to the validator committee from the validator set (i.e. from the set of staked users)
            validator_set = [a for a in state['agents']
                             if a.commitment_bond >= params['slash'].BOND_SIZE]
            ordered_validator_set = sorted(validator_set,
                                           key=lambda x: x.score,
                                           reverse=True)
            validator_committee = ordered_validator_set[:
                                                        params['stake'].VALIDATOR_COMMITTEE_SIZE]
            validator_committee_ids = [a.uuid for a in validator_committee]

            # For each slot in the epoch a sequencer/block proposer is drawn (based on score) from the validator committee
            proposer = validator_committee_ids[0]

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
                          validator_committee_ids,
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





def replace_suf(variable: str, default_value=0.0):
    """Creates replacing function for state update from string

    Args:
        variable (str): The variable name that is updated

    Returns:
        function: A function that continues the state across a substep
    """
    return lambda _1, _2, _3, state, signal: (
        variable,
        signal.get(variable, default_value),
    )


def add_suf(variable: str, default_value=0.0):
    """
    Creates replacing function for state update from string

    Args:
        variable (str): The variable name that is updated

    Returns:
        function: A function that continues the state across a substep
    """
    return lambda _1, _2, _3, state, signal: (
        variable,
        signal.get(variable, default_value) + state[variable],
    )