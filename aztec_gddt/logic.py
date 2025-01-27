from aztec_gddt.types import *
from copy import deepcopy, copy
from random import sample, random, uniform, normalvariate
import numpy as np
import scipy.stats as st
from aztec_gddt.types import Slot
from aztec_gddt.mechanism_functions import block_reward, compute_base_fee, expected_profit_per_tx, target_mana_per_block, proving_cost_fn
import math


def p_evolve_time(params: ModelParams, _2, _3, _4):
    return {'delta_l1_blocks': params['timestep_in_l1_blocks']}


def s_blocks_passed(_1, _2, _3,
                    state: ModelState,
                    signal):
    return ('l1_blocks_passed', state['l1_blocks_passed'] + signal['delta_l1_blocks'])


def s_delta_blocks(_1, _2, _3, _4, signal):
    return ('delta_l1_blocks', signal['delta_l1_blocks'])


def p_epoch(params: ModelParams, _2, history: list[list[ModelState]], state: ModelState):
    """
    Logic for the evolution over the epoch/slot state
    """
    last_epoch = state['last_epoch']
    epoch = deepcopy(state['current_epoch'])
    dropped_tx = 0
    excl_tx = 0
    total_tx = 0
    excess = state['excess_mana']
    l2_blocks_passed = 0
    base_fee = state['base_fee']

    # Interpret zero slots as a signal for creating a new Epoch
    if len(epoch.slots) == 0:
        pass
    else:
        curr_slot = epoch.slots[-1]

    l1_blocks_since_slot_init = state['l1_blocks_passed'] - \
        curr_slot.init_time_in_l1

    if l1_blocks_since_slot_init < params['L1_SLOTS_PER_L2_SLOT']:
        # If there's still slot time,
        # check whatever events have progressed
        if l1_blocks_since_slot_init >= curr_slot.time_until_E_BLOCK_SENT:
            curr_slot.has_block_header_on_l1 = True

        if l1_blocks_since_slot_init >= curr_slot.time_until_E_BLOCK_VALIDATE:
            curr_slot.has_validator_signatures = True

        if l1_blocks_since_slot_init >= curr_slot.time_until_E_BLOCK_PROPOSE:
            curr_slot.has_proposal_on_network = True

            # XXX consider using a random distribution
            curr_slot.tx_count = params['AVERAGE_TX_COUNT_PER_SLOT']
            total_tx += curr_slot.tx_count
            # # XXX consider adding a random term
            # curr_slot.tx_total_mana = curr_slot.tx_count * \
            #     params['general'].OVERHEAD_MANA_PER_TX
            # HACK

            # XXX: assume that base fee is computed when block is proposed
            # FIXME

            past_past_base_fee = history[-2][-1]['base_fee']
            past_base_fee = history[-1][-1]['base_fee']

            if ~np.isnan(past_past_base_fee):
                inflation_estimate = (
                    history[-1][-1]['base_fee'] / history[-2][-1]['base_fee']) - 1
            else:
                inflation_estimate = 1.0

            max_fee: JuicePerMana = (
                1 + inflation_estimate) * past_base_fee

            max_fee_avg = (
                1 + inflation_estimate * params['MAX_FEE_INFLATION_RELATIVE_MEAN']) * past_base_fee

            max_fee_std = params['MAX_FEE_INFLATION_RELATIVE_STD'] * max_fee

            max_fees = st.norm.rvs(loc=max_fee_avg,
                                   scale=max_fee_std,
                                   size=[total_tx])

            inds_valid_due_to_max_above_base = max_fees > base_fee

            inds_valid_due_to_profitability = expected_profit_per_tx(
                params, state, max_fees, 0.00, total_tx) > 0

            passively_excl_inds = np.bitwise_not(
                inds_valid_due_to_max_above_base)
            actively_excl_inds = np.bitwise_not(
                inds_valid_due_to_profitability) & np.bitwise_not(passively_excl_inds)

            valid_inds = np.bitwise_not(
                passively_excl_inds | actively_excl_inds)

            excl_tx = np.sum(passively_excl_inds)
            dropped_tx = np.sum(actively_excl_inds)  # type: ignore

            curr_slot.tx_total_mana = (
                total_tx - excl_tx - dropped_tx) * params['OVERHEAD_MANA_PER_TX']

            curr_slot.tx_total_fee = max_fees[valid_inds].sum()

    else:
        # If slot time has expired
        # then check whatever there's still
        # space-time on the epoch

        # Compute excess mana during this block
        spent = curr_slot.tx_total_mana
        excess = max(excess + spent - target_mana_per_block(params), 0)
        l2_blocks_passed += 1

        l1_blocks_since_epoch_init = state['l1_blocks_passed'] - \
            epoch.init_time_in_l1

        epoch_still_ongoing = (l1_blocks_since_epoch_init <=
                               params['L2_SLOTS_PER_L2_EPOCH']
                               * params['L1_SLOTS_PER_L2_SLOT'])

        epoch_still_has_slots = len(
            epoch.slots) < params['L2_SLOTS_PER_L2_EPOCH']

        # Move on to the next slot or epoch
        t1 = st.gamma.rvs(2, scale=1/5)
        t2 = st.gamma.rvs(2, scale=1/3)
        t3 = st.gamma.rvs(2, scale=1/10)

        i_slot = len(epoch.slots)
        if epoch_still_has_slots and epoch_still_ongoing:
            # If there's space-time on the epoch
            # create a new slot

            # For each slot in the epoch a sequencer/block proposer is drawn (based on score) from the validator committee
            proposer = epoch.validators[i_slot]

            # NOTE: slot is created here
            new_slot = Slot(state['l1_blocks_passed'],
                            proposer,
                            time_until_E_BLOCK_PROPOSE=int(t1),
                            time_until_E_BLOCK_VALIDATE=int(t1 + t2),
                            time_until_E_BLOCK_SENT=int(t1 + t2 + t3))

            epoch.slots.append(new_slot)
        else:
            # If there isn't space-time on the epoch
            # create new epoch and slot
            last_epoch = deepcopy(epoch)
            last_epoch.pending_time_in_l1 = state['l1_blocks_passed']

            # N validators are drawn (based on score) to the validator committee from the validator set (i.e. from the set of staked users)
            validator_set = [a for a in state['agents']
                             if a.commitment_bond >= params['BOND_SIZE']]
            ordered_validator_set = sorted(validator_set,
                                           key=lambda x: x.score,
                                           reverse=True)
            validator_committee = ordered_validator_set[:
                                                        params['VALIDATOR_COMMITTEE_SIZE']]
            validator_committee_ids = [a.uuid for a in validator_committee]

            # For each slot in the epoch a sequencer/block proposer is drawn (based on score) from the validator committee
            proposer = validator_committee_ids[0]

            # NOTE: slot is created here
            new_slot = Slot(state['l1_blocks_passed'],
                            proposer,
                            time_until_E_BLOCK_PROPOSE=int(t1),
                            time_until_E_BLOCK_VALIDATE=int(t1 + t2),
                            time_until_E_BLOCK_SENT=int(t1 + t2 + t3))

            t4 = st.geom.rvs(0.25)
            t5 = st.geom.rvs(0.15)

            # NOTE: epoch is created here
            epoch = Epoch(init_time_in_l1=state['l1_blocks_passed'],
                          validators=validator_committee_ids,
                          slots=[new_slot],
                          time_until_E_EPOCH_QUOTE_ACCEPT=int(t4),
                          time_until_E_EPOCH_FINISH=int(t4 + t5))

    return {'current_epoch': epoch,
            'last_epoch': last_epoch,
            'cumm_dropped_tx': dropped_tx,
            'cumm_excl_tx': excl_tx,
            'cumm_total_tx': total_tx,
            'excess_mana': excess,
            'l2_blocks_passed': l2_blocks_passed}


def p_pending_epoch_proof(params: ModelParams, _2, _3,
                          state: ModelState) -> dict:
    agents = state['agents']
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
                    drift_speed_adj=params['BLOCK_REWARD_SPEED_ADJ'],
                    drift_decay_rate=params['BLOCK_REWARD_DRIFT_DECAY_RATE'],
                    volatility_coefficient=params['BLOCK_REWARD_VOLATILITY'],
                    volatility_decay_rate=params['BLOCK_REWARD_DRIFT_DECAY_RATE'])

                last_reward_time = epoch.finalized_time_in_l1

                delta_resolved_epochs += 1
                delta_finalized_epochs += 1
                delta_cumm_mana += sum(s.tx_total_mana for s in epoch.slots)
                delta_finalized_blocks += len(
                    [s for s in epoch.slots if s.has_block_header_on_l1])
                delta_empty_blocks += len(
                    [s for s in epoch.slots if not s.has_block_header_on_l1])
                agents = deepcopy(agents)
                for a in agents:
                    a.score = random()
            else:
                # If prover didn't finalize, then
                if t > params['L2_SLOTS_PER_L2_EPOCH'] * params['L1_SLOTS_PER_L2_SLOT']:
                    # Reorg epoch and slash prover if time is over
                    epoch.reorged = True

                    delta_empty_blocks += len(epoch.slots)
                    delta_unproven_epochs += 1
                    delta_resolved_epochs += 1
                    agents = deepcopy(agents)
                    for a in agents:
                        a.score = random()

                else:
                    # Or just wait
                    pass
        else:
            # If there isn't an accepted prover, then
            if t < params['PROVER_SEARCH_PERIOD']:
                if t < epoch.time_until_E_EPOCH_QUOTE_ACCEPT:
                    # Create quotes while no one is accepted

                    # XXX
                    # Assume that each timestep
                    # a agent quotes between 0% and 50%
                    # FIXME
                    agent = sample(population=state['agents'], k=1)[0]
                    quote = uniform(0.0, 0.5)
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
                delta_empty_blocks += len(epoch.slots)
                delta_unproven_epochs += 1
                delta_resolved_epochs += 1
                agents = deepcopy(agents)
                for a in agents:
                    a.score = random()

    return {'last_epoch': epoch,
            'last_reward': last_reward,
            'last_reward_time_in_l1': last_reward_time,
            'cumm_empty_blocks': delta_empty_blocks,
            'cumm_unproven_epochs': delta_unproven_epochs,
            'cumm_resolved_epochs': delta_resolved_epochs,
            'cumm_finalized_epochs': delta_finalized_epochs,
            'cumm_mana_used_on_finalized_blocks': delta_cumm_mana,
            'cumm_finalized_blocks': delta_finalized_blocks,
            'agents': agents
            }


def s_congestion_multiplier(params: ModelParams, _2, _3, state: ModelState, signal) -> tuple:
    if state['timestep'] <= 1:
        multiplier = params['MINIMUM_MULTIPLIER_CONGESTION']
    else:
        upper_multiplier = state['congestion_multiplier'] * \
            (1 + params['MAX_RELATIVE_CHANGE_CONGESTION'])

        lower_multiplier = state['congestion_multiplier'] * \
            (1 - params['MAX_RELATIVE_CHANGE_CONGESTION'])

        update_frac = params['RELATIVE_UPDATE_FRACTION_CONGESTION'] * \
            params['MAXIMUM_MANA_PER_BLOCK']
        multiplier = params['MINIMUM_MULTIPLIER_CONGESTION']
        multiplier *= math.exp(state['excess_mana'] / update_frac)

        if multiplier > upper_multiplier:
            multiplier = upper_multiplier
        elif multiplier < lower_multiplier:
            multiplier = lower_multiplier

    return ('congestion_multiplier', multiplier)


def generic_oracle(var_real, var_oracle, var_update_time, max_param=''):
    def p_oracle_update(params: dict, _2, _3, state: dict) -> dict:

        now = state['l1_blocks_passed']
        value = state[var_oracle]
        update_time = state[var_update_time]

        cond1 = now > (update_time + params['MIN_ORACLE_UPDATE_LAG_C'])
        cond2 = random() < params['ORACLE_UPDATE_FREQUENCY_E']
        cond3 = state['timestep'] <= 1

        do_update = (cond1 & cond2) | cond3

        if do_update:
            if max_param == '':
                value = state[var_real]
            else:
                if state[var_real] > value * (1 + params[max_param]):
                    value = value * (1 + params[max_param])
                elif state[var_real] < value * (1 - params[max_param]):
                    value = value * (1 - params[max_param])
                else:
                    value = state[var_real]

            update_time = now

        return {var_oracle: value, var_update_time: update_time}
    return p_oracle_update


def generic_uniform_with_initial(state_var: str, param_initial_value: str):
    def p_oracle(params: dict, _2, _3, state: dict) -> dict:

        if state['timestep'] <= 1:
            value = params[param_initial_value]
        else:
            relative_change = uniform(-params['MAXIMUM_UPDATE_PERCENTAGE_C'],
                                      params['MAXIMUM_UPDATE_PERCENTAGE_C'])
            value = state[state_var] * (1 + relative_change)

        return {state_var: value}
    return p_oracle


def p_oracle_proving_cost(params: ModelParams, _2, _3, state: ModelState) -> dict:

    if state['timestep'] <= 1:
        MANA_PER_TX = params['RELATIVE_TARGET_MANA_PER_BLOCK'] * \
            params['MAXIMUM_MANA_PER_BLOCK'] / \
            params['AVERAGE_TX_COUNT_PER_SLOT']
        WEI_PER_USD = (10 ** 18) / params['market_price_eth']
        PROOF_COST_IN_WEI_PER_MANA = params['PROVING_COST_INITIAL_IN_USD_PER_TX_C'] * \
            WEI_PER_USD / MANA_PER_TX
    else:
        relative_change = uniform(-params['MAXIMUM_UPDATE_PERCENTAGE_C'],
                                  params['MAXIMUM_UPDATE_PERCENTAGE_C'])
        PROOF_COST_IN_WEI_PER_MANA = state['oracle_proving_cost'] * (
            1 + relative_change)

    return {'oracle_proving_cost': PROOF_COST_IN_WEI_PER_MANA}


p_oracle_juice_per_wei = generic_oracle(
    'market_price_juice_per_wei',
    'oracle_price_juice_per_wei',
    'update_time_oracle_price_juice_per_wei',
    'MAXIMUM_UPDATE_PERCENTAGE_C')

p_oracle_l1_gas = generic_oracle(
    'market_price_l1_gas',
    'oracle_price_l1_gas',
    'update_time_oracle_price_l1_gas')

p_oracle_l1_blobgas = generic_oracle(
    'market_price_l1_blobgas',
    'oracle_price_l1_blobgas',
    'update_time_oracle_price_l1_blobgas')


def generic_random_walk(var, mu, std, do_round=True):
    def s_random_walk(params: ModelParams, _2, _3, state: dict, signal) -> tuple:

        raw_value = max(state[var] + normalvariate(mu, std), 0)
        if do_round:
            value = round(raw_value)
        else:
            value = raw_value

        return (var, value)
    return s_random_walk


def generic_gaussian_noise(var,
                           mu_param,
                           cov_param,
                           do_round=True,
                           min_value=0.0,
                           max_rel_change=float('nan')):
    def s_random_walk(params, _2, _3, state: dict, signal) -> tuple:

        if state['timestep'] <= 1:
            raw_value = params[mu_param]
        else:
            raw_value = max(
                state[var] + normalvariate(0, params[mu_param] * params[cov_param]), min_value)

        if do_round:
            value = round(raw_value)  # type: ignore
        else:
            value = raw_value  # type: ignore


        if np.isfinite(max_rel_change):
            past_value = state[var]

            lower_bound = past_value * (1 - max_rel_change)
            upper_bound = past_value * (1 + max_rel_change)
            if value < lower_bound:
                value = lower_bound
            elif value > upper_bound:
                value = upper_bound
            else:
                pass
        else:
            pass

        return (var, value)
    return s_random_walk


s_market_price_juice_per_wei = generic_gaussian_noise(
    'market_price_juice_per_wei', 'JUICE_PER_WEI_MEAN', 'JUICE_PER_WEI_COV', False, min_value=0.0)

s_market_price_l1_gas = generic_gaussian_noise(
    'market_price_l1_gas', 'WEI_PER_L1GAS_MEAN', 'WEI_PER_L1GAS_COV', True, min_value=1.0, max_rel_change=0.125)
s_market_price_l1_blobgas = generic_gaussian_noise(
    'market_price_l1_blobgas', 'WEI_PER_L1BLOBGAS_MEAN', 'WEI_PER_L1BLOBGAS_COV', True, min_value=1.0, max_rel_change=0.125)


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
