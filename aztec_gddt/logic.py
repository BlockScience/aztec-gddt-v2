from aztec_gddt.types import *
from copy import deepcopy, copy
from random import sample, random
from aztec_gddt.types import Slot
from aztec_gddt.mechanism_functions import block_reward


def p_evolve_time(params: ModelParams, _2, _3, _4):
    return {'delta_l1_blocks': params['timestep_in_l1_blocks']}


def s_blocks_passed(_1, _2, _3,
                    state: ModelState,
                    signal):
    return ('l1_blocks_passed', state['l1_blocks_passed'] + signal['delta_l1_blocks'])


def s_delta_blocks(_1, _2, _3, _4, signal):
    return ('delta_l1_blocks', signal['delta_l1_blocks'])


def p_epoch(params: ModelParams, _2, _3, state: ModelState):
    """
    Logic for the evolution over the epoch/slot state
    """
    last_epoch = state['last_epoch']
    epoch = deepcopy(state['current_epoch'])

    # Interpret zero slots as a signal for creating a new Epoch
    if len(epoch.slots) == 0:
        pass

    else:
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

            # XXX consider using a random distribution
            curr_slot.tx_count = params['behavior'].AVERAGE_TX_COUNT_PER_SLOT
            # XXX consider adding a random term
            curr_slot.tx_total_mana = curr_slot.tx_count * \
                params['general'].OVERHEAD_MANA_PER_TX
    else:

        l1_blocks_since_epoch_init = state['l1_blocks_passed'] - \
            epoch.init_time_in_l1

        epoch_still_ongoing = (l1_blocks_since_epoch_init <=
                               params['general'].L2_SLOTS_PER_L2_EPOCH
                               * params['general'].L1_SLOTS_PER_L2_SLOT)

        epoch_still_has_slots = len(
            epoch.slots) < params['general'].L2_SLOTS_PER_L2_EPOCH

        # Move on to the next slot or epoch
        t1 = 0.75  # TODO
        t2 = 0.25  # TODO
        t3 = 0.5  # TODO

        i_slot = len(epoch.slots)
        if epoch_still_has_slots and epoch_still_ongoing:

            # For each slot in the epoch a sequencer/block proposer is drawn (based on score) from the validator committee
            proposer = epoch.validators[i_slot]

            # NOTE: slot is created here
            new_slot = Slot(state['l1_blocks_passed'],
                            proposer,
                            time_until_E_BLOCK_PROPOSE=t1,
                            time_until_E_BLOCK_VALIDATE=t1 + t2,
                            time_until_E_BLOCK_SENT=t1 + t2 + t3)

            epoch.slots.append(new_slot)
        else:
            last_epoch = deepcopy(epoch)
            last_epoch.pending_time_in_l1 = state['l1_blocks_passed']

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
                            time_until_E_BLOCK_PROPOSE=t1,
                            time_until_E_BLOCK_VALIDATE=t1 + t2,
                            time_until_E_BLOCK_SENT=t1 + t2 + t3)

            t4 = 5
            t5 = t4 + 1

            # NOTE: epoch is created here
            epoch = Epoch(init_time_in_l1=state['l1_blocks_passed'],
                          validators=validator_committee_ids,
                          slots=[new_slot],
                          time_until_E_EPOCH_QUOTE_ACCEPT=t4,
                          time_until_E_EPOCH_FINISH=t5)

    return {'current_epoch': epoch,
            'last_epoch': last_epoch}

    # Check if all blocks are done

    if curr_slot.is_valid_proposal:
        epoch.slots.append(new_slot)
        # TODO: create new slot

    # 1. Update slot state based on BlockEvolutionState
    # 2. If time is over or slot is pending-finish, create new block


def p_pending_epoch_proof(params: ModelParams, _2, _3,
                          state: ModelState) -> dict:
    epoch = state['last_epoch']
    last_reward_time = state['last_reward_time_in_l1']
    last_reward = state['last_reward']

    if epoch.finalized or epoch.reorged:
        pass
    else:
        epoch = deepcopy(state['last_epoch'])
        t = state['l1_blocks_passed'] - epoch.pending_time_in_l1

        if epoch.accepted_prover != None:
            if t > epoch.time_until_E_EPOCH_FINISH:
                # Finalize epoch and perform rewards
                epoch.finalized = True
                epoch.finalized_time_in_l1 = state['l1_blocks_passed']

                last_reward = block_reward(
                    curr_reward_time=state['l1_blocks_passed'],
                    prev_reward_time=last_reward_time,
                    prev_reward_value=last_reward,
                    drift_speed_adj=params['reward'].BLOCK_REWARD_SPEED_ADJ,
                    drift_decay_rate=params['reward'].BLOCK_REWARD_DRIFT_DECAY_RATE,
                    volatility_coefficient=params['reward'].BLOCK_REWARD_VOLATILITY,
                    volatility_decay_rate=params['reward'].BLOCK_REWARD_DRIFT_DECAY_RATE)
                last_reward_time = epoch.finalized_time_in_l1
            else:
                if t > params['general'].L2_SLOTS_PER_L2_EPOCH * params['general'].L1_SLOTS_PER_L2_SLOT:
                    # Reorg epoch and slash prover
                    epoch.reorged = True
                else:
                    pass
        else:
            if t < epoch.time_until_E_EPOCH_QUOTE_ACCEPT:
                # Generate quotes
                # XXX
                # Assume that each timestep
                # a random agent proposes a random percentage of his commit bond
                # up until 20%
                agent = sample(population=state['agents'], k=1)[0]
                quote = agent.commitment_bond * random() * 0.2
                epoch.prover_quotes[agent.uuid] = quote
            else:
                if len(epoch.prover_quotes) > 0:
                    # Select highest scoring quote
                    prover = min(epoch.prover_quotes,
                                 key=epoch.prover_quotes.get)  # type: ignore
                    epoch.accepted_prover = prover
                    epoch.accepted_prover_quote = epoch.prover_quotes[prover]
                else:
                    # Reorg
                    epoch.reorged = True

    return {'last_epoch': epoch,
            'last_reward': last_reward,
            'last_reward_time_in_l1': last_reward_time}


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
