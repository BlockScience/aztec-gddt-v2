from aztec_gddt.types import *
from copy import deepcopy, copy
from random import sample, random, uniform
from aztec_gddt.types import Slot
from aztec_gddt.mechanism_functions import block_reward
import math

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
    dropped_tx = 0
    excl_tx = 0
    excess = state['excess_mana']
    l2_blocks_passed = 0

    # Interpret zero slots as a signal for creating a new Epoch
    if len(epoch.slots) == 0:
        pass
    else:
        curr_slot = epoch.slots[-1]

    l1_blocks_since_slot_init = state['l1_blocks_passed'] - \
        curr_slot.init_time_in_l1

    if l1_blocks_since_slot_init < params['general'].L1_SLOTS_PER_L2_SLOT:
        # If there's still slot time,
        # check whatever events have progressed
        if l1_blocks_since_slot_init >= curr_slot.time_until_E_BLOCK_SENT:
            curr_slot.has_block_header_on_l1 = True

        if l1_blocks_since_slot_init >= curr_slot.time_until_E_BLOCK_VALIDATE:
            curr_slot.has_validator_signatures = True

        if l1_blocks_since_slot_init >= curr_slot.time_until_E_BLOCK_PROPOSE:
            curr_slot.has_proposal_on_network = True

            # XXX consider using a random distribution
            curr_slot.tx_count = params['behavior'].AVERAGE_TX_COUNT_PER_SLOT
            # # XXX consider adding a random term
            # curr_slot.tx_total_mana = curr_slot.tx_count * \
            #     params['general'].OVERHEAD_MANA_PER_TX
            # HACK
            curr_slot.tx_total_mana = int(params['general'].MAXIMUM_MANA_PER_BLOCK * uniform(0.45, 0.54))
    else:
        # If slot time has expired
        # then check whatever there's still
        # space-time on the epoch



        # Compute excess mana during this block
        spent = curr_slot.tx_total_mana
        excess = max(excess + spent - params['fee'].TARGET_MANA_PER_BLOCK, 0)
        l2_blocks_passed += 1


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
            # If there's space-time on the epoch
            # create a new slot

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
            # If there isn't space-time on the epoch
            # create new epoch and slot
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
            'last_epoch': last_epoch,
            'cumm_dropped_tx': dropped_tx,
            'cumm_excl_tx': excl_tx,
            'excess_mana': excess,
            'l2_blocks_passed': l2_blocks_passed}


def p_pending_epoch_proof(params: ModelParams, _2, _3,
                          state: ModelState) -> dict:
    epoch = state['last_epoch']
    last_reward_time = state['last_reward_time_in_l1']
    last_reward = state['last_reward']

    delta_empty_blocks = 0
    delta_unproven_epochs = 0
    delta_resolved_epochs = 0
    delta_finalized_epochs = 0
    delta_cumm_mana = 0
    delta_finalized_blocks = 0


    # Ignore resolved epochs (eg. finalized or reorged)
    if epoch.finalized or epoch.reorged:
        pass
    else:
        epoch = deepcopy(state['last_epoch'])
        t = state['l1_blocks_passed'] - epoch.pending_time_in_l1

        if epoch.accepted_prover != None:
            # If there's an accepted prover, then
            if t > epoch.time_until_E_EPOCH_FINISH:
                # If prover has finalized, then
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

                delta_resolved_epochs += 1
                delta_finalized_epochs +=1
                delta_cumm_mana += sum(s.tx_total_mana for s in epoch.slots)
                delta_finalized_blocks += len(epoch.slots)
            else:
                # If prover didn't finalize, then
                if t > params['general'].L2_SLOTS_PER_L2_EPOCH * params['general'].L1_SLOTS_PER_L2_SLOT:
                    # Reorg epoch and slash prover if time is over
                    epoch.reorged = True

                    # TODO Maybe we need to check slots individually
                    delta_empty_blocks += len(epoch.slots)
                    delta_unproven_epochs += 1
                    delta_resolved_epochs += 1

                else:
                    # Or just wait
                    pass
        else:
            # If there isn't an accepted prover, then
            if t < params['general'].PROVER_SEARCH_PERIOD:
                if t < epoch.time_until_E_EPOCH_QUOTE_ACCEPT:
                    # Create quotes while no one is accepted

                    # XXX
                    # Assume that each timestep
                    # a random agent proposes a random percentage of his commit bond
                    # up until 20%
                    # FIXME
                    agent = sample(population=state['agents'], k=1)[0]
                    quote = agent.commitment_bond * random() * 0.2
                    epoch.prover_quotes[agent.uuid] = quote
                else:
                    # If time for acceptance is over, then
                    if len(epoch.prover_quotes) > 0:
                        # Select highest scoring admissible quote
                        prover = min(epoch.prover_quotes,
                                    key=epoch.prover_quotes.get)  # type: ignore
                        epoch.accepted_prover = prover
                        epoch.accepted_prover_quote = epoch.prover_quotes[prover]
                    else:
                        # Else, keep waiting
                        pass
            else:
                # Or if time for accepting a prover is over, reorg
                epoch.reorged = True
                # TODO Maybe we need to check slots individually
                delta_empty_blocks += len(epoch.slots)
                delta_unproven_epochs += 1
                delta_resolved_epochs += 1

    return {'last_epoch': epoch,
            'last_reward': last_reward,
            'last_reward_time_in_l1': last_reward_time,
            'cumm_empty_blocks': delta_empty_blocks,
            'cumm_unproven_epochs': delta_unproven_epochs,
            'cumm_resolved_epochs': delta_resolved_epochs,
            'cumm_finalized_epochs': delta_finalized_epochs,
            'cumm_mana_used_on_finalized_blocks': delta_cumm_mana,
            'cumm_finalized_blocks': delta_finalized_blocks,
            }



def s_congestion_multiplier(params: ModelParams, _2, _3, state: ModelState, signal) -> tuple:

    upper_multiplier = state['congestion_multiplier'] * (1 + params['fee'].MAX_RELATIVE_CHANGE_CONGESTION)

    lower_multiplier = state['congestion_multiplier'] * (1 - params['fee'].MAX_RELATIVE_CHANGE_CONGESTION)

    multiplier = params['fee'].MINIMUM_MULTIPLIER_CONGESTION * math.exp(state['excess_mana'] / params['fee'].UPDATE_FRACTION_CONGESTION)

    if multiplier > upper_multiplier:
        multiplier = upper_multiplier
    elif multiplier < lower_multiplier:
        multiplier = lower_multiplier

    return ('congestion_multiplier', multiplier)



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

