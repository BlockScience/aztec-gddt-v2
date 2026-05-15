"""
Core simulation logic for the Aztec GDDT v2 model.

This module implements the simulation logic, including:
- Time evolution functions (p_evolve_time, s_blocks_passed)
- Epoch and slot lifecycle management (p_epoch)
- Proving and finalization logic (p_pending_epoch_proof)
- Oracle price update mechanisms
- Market price simulation functions

Function naming conventions:
- p_* functions: Policy functions that compute signals based on current state
- s_* functions: State update functions that apply signals to update state variables
- generic_* functions: Function factories that create specialized functions

This logic is executed through state update blocks in structure.py 
"""

from aztec_gddt.types import *
from copy import deepcopy, copy
from random import sample, random, uniform, normalvariate, choices, choice, randint
import numpy as np
import numpy.typing as npt
import scipy.stats as st
from aztec_gddt.types import Agent, Slot
from aztec_gddt.mechanism_functions import block_reward, expected_profit_per_tx, target_mana_per_block, proving_cost_fn
import math


def p_evolve_time(params: ModelParams, _2, _3, _4):
    """
    Policy function that advances time in the simulation.

    Generates a signal with the number of L1 blocks to advance based on the timestep parameter.

    Args:
        params: Model parameters containing timestep_in_l1_blocks
        _2, _3, _4: Unused parameters (history, state)

    Returns:
        dict: Signal with delta_l1_blocks indicating how many blocks to advance
    """
    return {'delta_l1_blocks': params['timestep_in_l1_blocks']}


def s_blocks_passed(_1, _2, _3,
                    state: ModelState,
                    signal):
    """
    State update function that increments the total L1 blocks count.

    Args:
        _1, _2, _3: Unused parameters
        state: Current model state
        signal: Signal containing delta_l1_blocks

    Returns:
        tuple: Key-value pair to update in the state (l1_blocks_passed)
    """
    return ('l1_blocks_passed', state['l1_blocks_passed'] + signal['delta_l1_blocks'])


def s_delta_blocks(_1, _2, _3, _4, signal):
    """
    State update function that passes through the delta_l1_blocks signal.

    Args:
        _1, _2, _3, _4: Unused parameters
        signal: Signal containing delta_l1_blocks

    Returns:
        tuple: Key-value pair to update in the state (delta_l1_blocks)
    """
    return ('delta_l1_blocks', signal['delta_l1_blocks'])


def p_epoch(params: ModelParams, _2, history: list[list[ModelState]], state: ModelState):
    """
    Evolves the epoch and slot state in the simulation.

    This is a core function that:
    1. Processes current slot lifecycle events (proposal, validation, signature collection)
    2. Creates new slots when current slots are completed
    3. Creates new epochs when current epochs are completed
    4. Handles transaction inclusion, exclusion, and fee calculations
    5. Tracks metrics like dropped transactions and block signatures

    Args:
        params: Model parameters for controlling slot and epoch behavior
        _2: Unused parameter
        history: History of previous states
        state: Current model state

    Returns:
        dict: Updated epoch state and associated metrics
    """
    # Initialize with current state and metrics
    last_epoch = state['last_epoch']
    epoch = deepcopy(state['current_epoch'])
    dropped_tx = 0
    excl_tx = 0
    total_tx = 0
    excess = state['excess_mana']
    l2_blocks_passed = 0
    base_fee = state['base_fee']

    # Track signature collection metrics for this step
    delta_blocks_with_collected_signatures = 0
    delta_blocks_with_enough_signatures = 0

    # Interpret a epoch with zero slots as one on which the current slot
    # is one with zero mana / no valid proposer / initializated at an ancient time
    # We don't expect the if-clause to be activated, and this serves
    # as a fail-safe.
    if len(epoch.slots) == 0:
        curr_slot = Slot(init_time_in_l1=-9999,
                         proposer='null',
                         time_until_E_BLOCK_PROPOSE=-9999,
                         time_until_E_BLOCK_VALIDATE=-9999,
                         time_until_E_BLOCK_SENT=-9999)
    else:
        # Get current slot (the last one in the epoch)
        curr_slot = epoch.slots[-1]

    # Calculate how many L1 blocks have passed since this slot was initialized
    l1_blocks_since_slot_init = state['l1_blocks_passed'] - \
        curr_slot.init_time_in_l1

    # CASE 1: Current slot still has time remaining and the current epoch has not been reorged
    if (l1_blocks_since_slot_init < params['L1_SLOTS_PER_L2_SLOT']) and (epoch.reorged_full == False):
        # Process possible events that may have occurred within slot's lifetime

        # Check if block header has been sent to L1
        if (l1_blocks_since_slot_init >= curr_slot.time_until_E_BLOCK_SENT):
            curr_slot.has_block_header_on_l1 = True

        # Check if validators have validated the block
        if (l1_blocks_since_slot_init >= curr_slot.time_until_E_BLOCK_VALIDATE) and curr_slot.has_block_header_on_l1 and not curr_slot.has_collected_signatures:
            # Determine validation signature count using binomial distribution
            # (some validators may skip signing with probability SIGNATURE_SKIP_PROBABILITY)
            max_signatures = params['VALIDATOR_COMMITTEE_SIZE']
            required_signatures = params['VALIDATOR_COMMITTEE_SIZE'] * \
                params['SIGNATURES_NEEDED'] + 1
            collected_signatures = st.binom.rvs(
                n=max_signatures, p=(1-params['SIGNATURE_SKIP_PROBABILITY']))
            curr_slot.has_collected_signatures = True
            delta_blocks_with_collected_signatures = 1

            # Check if enough signatures were collected to get over threshold
            if collected_signatures >= required_signatures:
                curr_slot.has_validator_signatures = True
                delta_blocks_with_enough_signatures = 1

        # Check if block can be proposed (only if validators have approved)
        if (l1_blocks_since_slot_init >= curr_slot.time_until_E_BLOCK_PROPOSE) and curr_slot.has_validator_signatures and not curr_slot.has_proposal_on_network:

            # Only propose if L1 gas price is below gas price threshold (can be used for Sequencer profitability decisions and censoring)
            if not (state['market_price_l1_gas'] > params['SEQUENCER_L1_GAS_PRICE_THRESHOLD_E']):

                # Calculate expected transactions and fee
                expected_total_tx = params['AVERAGE_TX_COUNT_PER_SLOT']

                # NOTE: assume that base fee is computed when block is proposed

                # Estimate (user expected) base fee inflation based on recent history
                past_past_base_fee = history[-2][-1]['base_fee']
                past_base_fee = history[-1][-1]['base_fee']

                if ~np.isnan(past_past_base_fee):
                    # Calculate inflation based on recent history
                    inflation_estimate = (
                        history[-1][-1]['base_fee'] / history[-2][-1]['base_fee']) - 1
                else:
                    # Use default inflation estimate for initial steps
                    inflation_estimate = params['INITIAL_INFLATION_ESTIMATE']

                 # Calculate max fee using inflation estimate
                max_fee: JuicePerMana = (
                    1 + inflation_estimate) * past_base_fee

                # Determine average and std dev for fee distribution
                max_fee_avg = max((
                    1 + inflation_estimate * params['MAX_FEE_INFLATION_RELATIVE_MEAN']) * past_base_fee, 0.0)
                max_fee_std = max(
                    params['MAX_FEE_INFLATION_RELATIVE_STD'] * max_fee, 0.0)

                # Generate random max fees for each transaction
                max_fees: npt.NDArray = np.array(st.norm.rvs(loc=max_fee_avg,
                                                             scale=max_fee_std,
                                                             size=[expected_total_tx]))

                # Determine which transactions pass fee validation
                # Two main criteria for transaction inclusion:
                # 1. Max fee > base fee (passive filter)
                # 2. Transaction is profitable (active filter)
                inds_valid_due_to_max_above_base = max_fees > base_fee

                inds_valid_due_to_profitability = expected_profit_per_tx(
                    params, state, max_fees, 0.00, expected_total_tx) > 0

                # Track excluded transactions by criteria
                passively_excl_inds = np.bitwise_not(
                    inds_valid_due_to_max_above_base)
                actively_excl_inds = np.bitwise_not(
                    inds_valid_due_to_profitability) & np.bitwise_not(passively_excl_inds)

                # Calculate indices of valid transactions (those not excluded)
                valid_inds = np.bitwise_not(
                    passively_excl_inds | actively_excl_inds)

                # Count transactions by status
                expected_excl_tx = np.sum(passively_excl_inds)
                expected_dropped_tx = np.sum(
                    actively_excl_inds)  # type: ignore

                expected_incl_tx = (expected_total_tx -
                                    expected_excl_tx - expected_dropped_tx)

                # Calculate block mana and fees based on included transactions
                expected_block_mana = (expected_incl_tx
                                       * params['OVERHEAD_MANA_PER_TX']
                                       * params['TOTAL_MANA_MULTIPLIER_E'])

                # type: ignore
                expected_block_fees = max_fees[valid_inds].sum()

                # Two scenarios: block fits within mana limit or needs to be capped
                if expected_block_mana <= params['MAXIMUM_MANA_PER_BLOCK']:
                    # Block fits within mana limits, use all values as calculated
                    curr_slot.has_proposal_on_network = True
                    curr_slot.tx_total_mana = expected_block_mana
                    curr_slot.tx_total_fee = expected_block_fees
                    total_tx = expected_total_tx
                    excl_tx = expected_excl_tx
                    dropped_tx = expected_dropped_tx
                else:
                    # Block exceeds mana limit, scale down proportionally
                    curr_slot.has_proposal_on_network = True
                    curr_slot.tx_total_mana = params['MAXIMUM_MANA_PER_BLOCK']
                    # Scale fees proportionally to mana reduction
                    curr_slot.tx_total_fee = expected_block_fees * \
                        params['MAXIMUM_MANA_PER_BLOCK'] / expected_block_mana
                    excl_tx = expected_excl_tx
                    # Scale included tx count proportionally
                    total_tx = int(
                        expected_incl_tx * params['MAXIMUM_MANA_PER_BLOCK'] / expected_block_mana)
                    dropped_tx = expected_total_tx - (total_tx + excl_tx)

                curr_slot.tx_count = total_tx

      # CASE 2: Current slot time has expired - proceed to next slot or epoch
    else:
        # If slot time has expired
        # then check whatever there's still
        # space-time on the epoch

        # Compute excess mana based on target mana value
        spent = curr_slot.tx_total_mana
        excess = max(excess + spent - target_mana_per_block(params), 0)
        l2_blocks_passed += 1

        # Check if current epoch has more slots (based on L1 blocks passed)
        l1_blocks_since_epoch_init = state['l1_blocks_passed'] - \
            epoch.init_time_in_l1

        # Determine if epoch is still ongoing
        epoch_still_ongoing = (l1_blocks_since_epoch_init <=
                               params['L2_SLOTS_PER_L2_EPOCH']
                               * params['L1_SLOTS_PER_L2_SLOT'])

        epoch_still_has_slots = len(
            epoch.slots) < params['L2_SLOTS_PER_L2_EPOCH']
        

        epoch_not_reorged = (epoch.reorged_full == False)

        # Move on to the next slot or epoch
        # Generate random timing parameters for the next slot
        # These determine when various events happen within the slot
        # Time until proposal must be gossiped through L2
        time_until_E_BLOCK_PROPOSE = st.gamma.rvs(
            params['SHAPE_E_BLOCK_PROPOSE'], scale=params['SCALE_E_BLOCK_PROPOSE'])

        # Time until validation (signatures) must be done
        time_until_E_BLOCK_VALIDATE = st.gamma.rvs(
            params['SHAPE_E_BLOCK_VALIDATE'], scale=params['SCALE_E_BLOCK_VALIDATE'])
        time_until_E_BLOCK_VALIDATE += time_until_E_BLOCK_PROPOSE

        # Time until header must be on L1
        time_until_E_BLOCK_SENT = st.gamma.rvs(
            params['SHAPE_E_BLOCK_SENT'], scale=params['SCALE_E_BLOCK_SENT'])
        time_until_E_BLOCK_SENT += time_until_E_BLOCK_VALIDATE

        i_slot = len(epoch.slots)
        # CASE 2A: Create a new slot within the current epoch
        if epoch_still_has_slots and epoch_still_ongoing and epoch_not_reorged:
            # If there's space-time on the epoch
            # create a new slot

            # For each slot in the epoch a sequencer/block proposer is drawn (based on score) from the validator committee
            proposer = epoch.validators[i_slot]

            # Create new slot with timing parameters
            new_slot = Slot(state['l1_blocks_passed'],
                            proposer,
                            time_until_E_BLOCK_PROPOSE=int(
                                time_until_E_BLOCK_PROPOSE),
                            time_until_E_BLOCK_VALIDATE=int(
                                time_until_E_BLOCK_VALIDATE),
                            time_until_E_BLOCK_SENT=int(time_until_E_BLOCK_SENT))

            epoch.slots.append(new_slot)

        # CASE 2B: Create a new epoch (and its first slot)
        else:
            # Mark current epoch as last_epoch and mark it as pending
            last_epoch = deepcopy(epoch)
            last_epoch.pending_time_in_l1 = state['l1_blocks_passed']

            # N validators are drawn (based on random score) to the validator committee from the validator set (i.e. from the set of staked users)
            validator_set = [a for k, a in state['agents'].items()
                             if a.stake >= params['MIN_STAKE']
                             and k != 'prover']
            # Sort by score to select highest scoring validators
            ordered_validator_set = sorted(validator_set,
                                           key=lambda x: x.score,
                                           reverse=True)
            validator_committee = ordered_validator_set[:
                                                        params['VALIDATOR_COMMITTEE_SIZE']]
            validator_committee_ids = [a.uuid for a in validator_committee]

            # For each slot in the epoch a sequencer/block proposer is drawn (based on score) from the validator committee
            proposer = validator_committee_ids[0]

            # Create the first slot of the new epoch
            new_slot = Slot(state['l1_blocks_passed'],
                            proposer,
                            time_until_E_BLOCK_PROPOSE=int(
                                time_until_E_BLOCK_PROPOSE),
                            time_until_E_BLOCK_VALIDATE=int(
                                time_until_E_BLOCK_VALIDATE),
                            time_until_E_BLOCK_SENT=int(time_until_E_BLOCK_SENT))

            # Generate random timing parameters for epoch-level events

            # Time until epoch must be finalized
            # Under the multi-proof scheme, this happens after a entire epoch evolution
            # Eg. Duration = L2 Blocks per Epoch * L1 Blocks per L2 Block.
            time_until_E_EPOCH_FINISH = params['L2_SLOTS_PER_L2_EPOCH'] * \
                params['L1_SLOTS_PER_L2_SLOT']

            # Create new epoch structure
            epoch = Epoch(init_time_in_l1=state['l1_blocks_passed'],
                          validators=validator_committee_ids,
                          slots=[new_slot],
                          time_until_E_EPOCH_FINISH=int(time_until_E_EPOCH_FINISH))

    # Return updated state values and metrics
    return {'current_epoch': epoch,
            'last_epoch': last_epoch,
            'cumm_dropped_tx': dropped_tx,
            'cumm_excl_tx': excl_tx,
            'cumm_total_tx': total_tx,
            'excess_mana': excess,
            'l2_blocks_passed': l2_blocks_passed,
            'cumm_blocks_with_collected_signatures': delta_blocks_with_collected_signatures,
            'cumm_blocks_with_enough_signatures': delta_blocks_with_enough_signatures}


def p_pending_epoch_proof(params: ModelParams, _2, _3,
                          state: ModelState) -> dict:
    """
    Handles proving and finalization of pending epochs ("pending chain").

    This function handles:
    2. Epoch finalization when a prover completes the proof
    3. Epoch reorg when proofs are not submitted or fail
    4. Slashing of validators and provers when appropriate
    5. Block reward calculation and distribution

    Args:
        params: Model parameters for controlling proving behavior
        _2, _3: Unused parameters
        state: Current model state

    Returns:
        dict: Updated epoch state, reward info, and finalization metrics
    """
    agents = state['agents']
    curr_epoch = deepcopy(state['current_epoch'])
    last_epoch = deepcopy(state['last_epoch'])
    last_reward_time = state['last_reward_time_in_l1']
    last_reward = state['last_reward']
    last_reward_to_provers = state['last_reward_to_provers']
    last_reward_per_prover = state['last_reward_per_prover']
    slash_count = 0
    slash_amount = 0.0

    # Initialize metrics to track during this step
    delta_empty_blocks = 0
    delta_unproven_epochs = 0
    delta_resolved_epochs = 0
    delta_finalized_epochs = 0
    delta_finalized_blocks = 0
    delta_cumm_mana = 0
    delta_cumm_fee_paid = 0
    delta_cumm_fee_burnt = 0
    delta_cumm_fee_to_provers = 0
    delta_cumm_fee_to_sequencers = 0
    # Part 1: Submit proofs to the current (eg. under proposal)
    # and last (eg. pending) epochs.
    N_proofs_to_submit_last_epoch = int(st.poisson.rvs(params['EXPECTED_PROOFS_LAST_EPOCH_PER_TS'])) 
    N_proofs_to_submit_current_epoch: int = int(st.poisson.rvs(params['EXPECTED_PROOFS_CURRENT_EPOCH_PER_TS'])) 

    N_proofs_to_submit_last_epoch_complete = int(
        uniform(params['MIN_FRACTION_COMPLETE_PROOF_LAST_EPOCH'], params['MAX_FRACTION_COMPLETE_PROOF_LAST_EPOCH']) * N_proofs_to_submit_last_epoch) 
    N_proofs_to_submit_last_epoch_partial: int = N_proofs_to_submit_last_epoch - \
        N_proofs_to_submit_last_epoch_complete

    for i in range(N_proofs_to_submit_last_epoch_complete):
        uuids_to_select = tuple(agents.keys())
        proof_blocks = params['L2_SLOTS_PER_L2_EPOCH']
        proof = (choice(uuids_to_select), proof_blocks)
        last_epoch.submitted_proofs.append(proof)

    for i in range(N_proofs_to_submit_last_epoch_partial):
        uuids_to_select = tuple(agents.keys())
        proof_blocks = randint(1, params['L2_SLOTS_PER_L2_EPOCH'])
        proof = (choice(uuids_to_select), proof_blocks)
        last_epoch.submitted_proofs.append(proof)

    for i in range(N_proofs_to_submit_current_epoch):
        uuids_to_select = tuple(agents.keys())
        proof_blocks = randint(1, len(curr_epoch.slots))
        proof = (choice(uuids_to_select), proof_blocks)
        curr_epoch.submitted_proofs.append(proof)

    # Part 2: Finalize and/or reorg the current and last epochs.

    t = state['l1_blocks_passed'] - last_epoch.pending_time_in_l1
    blocks_to_slash = []

    if t >= (params['L1_SLOTS_PER_L2_SLOT'] * params['L2_SLOTS_PER_L2_EPOCH'] - 1):
        if len(last_epoch.submitted_proofs) > 0:
            # Case 1: 
            longest_chain = max(s
                                for _, s in last_epoch.submitted_proofs)

            accepted_provers = set(a
                                   for (a, s) in last_epoch.submitted_proofs
                                   if s == longest_chain)
            
            last_epoch.accepted_provers = accepted_provers
            

            if longest_chain < params['L2_SLOTS_PER_L2_EPOCH']:
                # Do a partial reorg on the last epoch
                # and a full reorg on the current epoch
                last_epoch.reorged_partial = True
                blocks_to_slash = last_epoch.slots[longest_chain:]
                last_epoch.slots = last_epoch.slots[:longest_chain]
                curr_epoch.reorged_full = True
                delta_empty_blocks += len(curr_epoch.slots)
                delta_resolved_epochs += 1


            # Compute Rewards
            # Calculate block reward using drift-decay model, see spec or market design docs for more details
            last_reward = block_reward(
                curr_reward_time=state['l1_blocks_passed'],
                prev_reward_time=last_reward_time,
                prev_reward_value=last_reward,
                drift_speed_adj=params['BLOCK_REWARD_SPEED_ADJ'],
                drift_decay_rate=params['BLOCK_REWARD_DRIFT_DECAY_RATE'],
                volatility_coefficient=params['BLOCK_REWARD_VOLATILITY'],
                volatility_decay_rate=params['BLOCK_REWARD_DRIFT_DECAY_RATE'])
            

            last_reward_to_provers = last_reward * params['BLOCK_REWARD_SHARE_PROVER']
            last_reward_per_prover = last_reward_to_provers / len(last_epoch.accepted_provers)

            last_reward_time = last_epoch.finalized_time_in_l1

            last_epoch.finalized = True
            delta_empty_blocks += longest_chain - len(last_epoch.slots)
            delta_resolved_epochs += 1
            delta_finalized_epochs += 1
            delta_cumm_mana += sum(s.tx_total_mana for s in last_epoch.slots)
            delta_cumm_fee_paid = delta_cumm_mana * state['base_fee']

            burnt_factor = (state['base_fee'] - state['base_fee_congestionless']) / state['base_fee']
            delta_cumm_fee_burnt = delta_cumm_fee_paid * burnt_factor

            prover_factor = (state['base_fee'] - state['base_fee_proverless']) / state['base_fee']
            delta_cumm_fee_to_provers += delta_cumm_fee_paid * prover_factor

            delta_cumm_fee_to_sequencers += (delta_cumm_fee_paid - delta_cumm_fee_burnt - delta_cumm_fee_to_provers)

            delta_finalized_blocks += longest_chain
                
        else:
            # Do a full reorg on the last and current epoch
            last_epoch.reorged_full = True
            blocks_to_slash = last_epoch.slots
            last_epoch.slots = []
            delta_empty_blocks += len(last_epoch.slots)
            delta_unproven_epochs += 1
            delta_resolved_epochs += 1

            curr_epoch.reorged_full = True
            delta_empty_blocks += len(curr_epoch.slots)
            delta_resolved_epochs += 1

        # Slash sequencers and validators on non-proved blocks.
        if last_epoch.reorged_full or last_epoch.reorged_partial:
            for block in blocks_to_slash:
                # Slash sequencer
                sequencer_slash_value = agents[block.proposer].stake * params['PERCENTAGE_STAKE_SLASHED_C']
                agents[block.proposer].stake -= sequencer_slash_value
                slash_amount += sequencer_slash_value
                slash_count += 1


                # # Slash validators
                # slash_count = int(
                #     random() * params['MAX_VALIDATORS_TO_SLASH'])
                # slashed_validators = sample(
                #     last_epoch.validators, slash_count)
                # for k in slashed_validators:
                #     slash_amount += agents[k].stake
                #     agents[k].stake = 0.0
            


    else:
        # Do nothing
        pass

    # Return updated state and metrics
    return {'current_epoch': curr_epoch,
            'last_epoch': last_epoch,
            'last_reward': last_reward,
            'last_reward_time_in_l1': last_reward_time,
            'last_reward_to_provers': last_reward_to_provers,
            'last_reward_per_prover': last_reward_per_prover,
            'cumm_empty_blocks': delta_empty_blocks,
            'cumm_unproven_epochs': delta_unproven_epochs,
            'cumm_resolved_epochs': delta_resolved_epochs,
            'cumm_finalized_epochs': delta_finalized_epochs,
            'cumm_mana_used_on_finalized_blocks': delta_cumm_mana,
            'cumm_finalized_blocks': delta_finalized_blocks,
            'cumm_fee_paid': delta_cumm_fee_paid,
            'cumm_fee_burnt': delta_cumm_fee_burnt,
            'cumm_fee_to_provers': delta_cumm_fee_to_provers,
            'cumm_fee_to_sequencers': delta_cumm_fee_to_sequencers,
            'agents': agents,
            'slash_count': slash_count,
            'slash_amount': slash_amount
            }


def s_congestion_multiplier(params: ModelParams, _2, _3, state: ModelState, signal) -> tuple:
    """
    State update function that calculates the congestion multiplier based on excess mana.

    The congestion multiplier adjusts the base fee based on network load and is bounded by
    minimum and maximum values and rate of change constraints.

    Args:
        params: Model parameters controlling congestion behavior
        _2, _3: Unused parameters
        state: Current model state
        signal: Input signal (unused)

    Returns:
        tuple: Key-value pair to update in the state (congestion_multiplier)
    """
    # Initialize with minimum multiplier on first step
    if state['timestep'] <= 1:
        multiplier = params['MINIMUM_MULTIPLIER_CONGESTION']
    else:
        # Calculate bounds on how much multiplier can change in one step
        upper_multiplier = state['congestion_multiplier'] * \
            (1 + params['MAX_RELATIVE_CHANGE_CONGESTION'])

        lower_multiplier = state['congestion_multiplier'] * \
            (1 - params['MAX_RELATIVE_CHANGE_CONGESTION'])

        # Calculate update fraction for congestion (scaling parameter)
        update_frac = params['RELATIVE_UPDATE_FRACTION_CONGESTION'] * \
            params['MAXIMUM_MANA_PER_BLOCK']

        # Start with minimum multiplier
        multiplier = params['MINIMUM_MULTIPLIER_CONGESTION']

        # Calculate ratio of excess mana to update fraction
        raw_ratio = state['excess_mana'] / update_frac
        max_ratio = params['MAXIMUM_MULTIPLIER_CONGESTION_RATIO']

        # Cap the ratio if it exceeds maximum
        if raw_ratio > max_ratio:
            effective_ratio = max_ratio
        else:
            effective_ratio = raw_ratio

        # Apply exponential scaling to congestion multiplier
        # This means fee increases exponentially with congestion
        multiplier *= math.exp(effective_ratio)

        # Apply bounds to ensure multiplier doesn't change too quickly
        if multiplier > upper_multiplier:
            multiplier = upper_multiplier
        elif multiplier < lower_multiplier:
            multiplier = lower_multiplier

    return ('congestion_multiplier', multiplier)


def generic_oracle(var_real,
                   var_oracle,
                   var_update_time,
                   max_param='',
                   lagged=True):
    """
    Function factory that creates oracle update functions.

    These oracle functions periodically update an oracle value based on the real value,
    with optional lag and maximum update constraints, conditional on agent behavior.

    Args:
        var_real: State variable name for the actual value
        var_oracle: State variable name for the oracle value
        var_update_time: State variable name for tracking last update time
        max_param: Parameter name for maximum allowed update percentage (optional)
        lagged: Whether updates should respect a minimum lag time

    Returns:
        function: A process function for updating the oracle
    """
    def p_oracle_update(params: dict, _2, _3, state: dict) -> dict:

        now = state['l1_blocks_passed']
        value = state[var_oracle]
        update_time = state[var_update_time]

        if lagged:
            cond1 = now > (update_time + params['MIN_ORACLE_UPDATE_LAG_C'])
        else:
            cond1 = True
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
    """
    Function factory that creates a stochastic update function.

    Creates a process function that updates a state variable with a uniformly distributed
    relative change, using an initial value from parameters.

    Args:
        state_var: State variable name to update
        param_initial_value: Parameter name for the initial value

    Returns:
        function: A process function for stochastic updates
    """
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
    """
    Process function that updates the proving cost oracle.

    Adjusts the proving cost based on initial parameters or applies
    stochastic updates to existing values.

    Args:
        params: Model parameters controlling proving costs
        _2, _3: Unused parameters
        state: Current model state

    Returns:
        dict: Updated oracle_proving_cost signal
    """

    # Initial setup on first timestep
    if state['timestep'] <= 1:
        # Calculate mana per transaction based on target block utilization
        MANA_PER_TX = params['RELATIVE_TARGET_MANA_PER_BLOCK'] * \
            params['MAXIMUM_MANA_PER_BLOCK'] / \
            params['AVERAGE_TX_COUNT_PER_SLOT']

        # Convert USD to Gwei based on ETH price
        GWEI_PER_USD = (10 ** 9) / params['market_price_eth']

        # Calculate proving cost in Gwei per mana unit
        PROOF_COST_IN_GWEI_PER_MANA = params['PROVING_COST_INITIAL_IN_USD_PER_TX_C'] * \
            GWEI_PER_USD / MANA_PER_TX
        
        PROOF_COST_IN_GWEI_PER_MANA *= params['M_PROVER_TARGET']
    else:
        PROOF_COST_IN_GWEI_PER_MANA = state['oracle_proving_cost']


    return {'oracle_proving_cost': PROOF_COST_IN_GWEI_PER_MANA}


p_oracle_juice_per_gwei = generic_oracle(
    var_real='market_price_juice_per_gwei',
    var_oracle='oracle_price_juice_per_gwei',
    var_update_time='update_time_oracle_price_juice_per_gwei',
    max_param='MAXIMUM_UPDATE_PERCENTAGE_C',
    lagged=False)

p_oracle_l1_gas = generic_oracle(
    var_real='market_price_l1_gas',
    var_oracle='oracle_price_l1_gas',
    var_update_time='update_time_oracle_price_l1_gas',
    lagged=True)

p_oracle_l1_blobgas = generic_oracle(
    var_real='market_price_l1_blobgas',
    var_oracle='oracle_price_l1_blobgas',
    var_update_time='update_time_oracle_price_l1_blobgas',
    lagged=True)


def generic_random_walk(var, mu, std, do_round=True):
    """
    Function factory that creates a random walk state update.

    Creates a state update function that applies a random walk to a state variable using a normal distribution for step sizes.

    Args:
        var: State variable name to update
        mu: Mean value for the normal distribution
        std: Standard deviation for the normal distribution
        do_round: Whether to round the value (for integer variables)

    Returns:
        function: State update function for random walk
    """
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
                           max_rel_change=float('nan'),
                           modification_key=None):
    """
    Function factory that creates a Gaussian noise-based state update function.

    Creates a state update function that adds Gaussian noise to a state variable, with constraints on minimum values and maximum relative changes.

    Args:
        var: State variable name to update
        mu_param: Parameter name for the mean value
        cov_param: Parameter name for the coefficient of variation
        do_round: Whether to round the value (for integer variables)
        min_value: Minimum allowed value
        max_rel_change: Maximum allowed relative change
        modification_key: Parameter name for an additional multiplier

    Returns:
        function: A state update function for Gaussian noise processes
    """
    def s_random_walk(params: ModelParams, _2, _3, state: ModelState, signal) -> tuple:

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

        if modification_key != None:
            value = value * (1 + params[modification_key])

        return (var, value)
    return s_random_walk


def s_market_price_juice_per_gwei(params: ModelParams, _2, _3, state: ModelState, signal) -> tuple:
    """
    State update function for the Juice/Gwei market exchange rate based on the configured scenario.

    Handles different scenarios for price evolution:
    - Stochastic: Random Gaussian updates
    - Constant: Fixed price
    - StrictlyIncreasing/StrictlyDecreasing: Linear price changes

    Args:
        params: Model parameters controlling price behavior
        _2, _3: Unused parameters
        state: Current model state
        signal: Input signal (unused)

    Returns:
        tuple: Key-value pair to update in the state (market_price_juice_per_gwei)
    """
    # Handle different price evolution scenarios
    if (params['JUICE_PER_GWEI_SCENARIO'] == JuiceGweiExchangeRateScenario.Stochastic):
        # Use Gaussian noise model for stochastic scenario
        (_, value) = generic_gaussian_noise(
            var='market_price_juice_per_gwei',
            mu_param='JUICE_PER_GWEI_MEAN',
            cov_param='JUICE_PER_GWEI_COV',
            do_round=False,
            min_value=0.0,
            modification_key='FEE_JUICE_PRICE_MODIFICATION_E')(params, _2, _3, state, signal)
    elif params['JUICE_PER_GWEI_SCENARIO'] == JuiceGweiExchangeRateScenario.Constant:
        # Use fixed value for constant scenario
        value = params['JUICE_PER_GWEI_MEAN']
    else:
        # For increasing/decreasing scenarios, calculate linear change over time

        # Calculate total range of price change based on coefficient of variation
        # The constant 0.28867... is related to triangular distribution properties
        dy = params['JUICE_PER_GWEI_COV']
        dy *= params['JUICE_PER_GWEI_MEAN']
        dy /= 0.28867802136059275

        # Define min and max values for the price range
        max_y = params['JUICE_PER_GWEI_MEAN'] + dy / 2
        min_y = params['JUICE_PER_GWEI_MEAN'] - dy / 2

        # Calculate step size for each timestep
        dy_per_ts = dy / params['N_timesteps']
        if params['JUICE_PER_GWEI_SCENARIO'] == JuiceGweiExchangeRateScenario.StrictlyIncreasing:
            # Linear increase from min to max over simulation time
            value = min_y + dy_per_ts * state['timestep']
        elif params['JUICE_PER_GWEI_SCENARIO'] == JuiceGweiExchangeRateScenario.StrictlyDecreasing:
            # Linear decrease from max to min over simulation time
            value = max_y - dy_per_ts * state['timestep']
        else:
            # Should never get here (invalid scenario)
            value = float('nan')

    return ('market_price_juice_per_gwei', value)


s_market_price_l1_gas = generic_gaussian_noise(
    'market_price_l1_gas', 'GWEI_PER_L1GAS_MEAN', 'GWEI_PER_L1GAS_COV', True, min_value=1.0, max_rel_change=0.125)
s_market_price_l1_blobgas = generic_gaussian_noise(
    'market_price_l1_blobgas', 'GWEI_PER_L1BLOBGAS_MEAN', 'GWEI_PER_L1BLOBGAS_COV', True, min_value=1.0, max_rel_change=0.125)


def replace_suf(variable: str, default_value=0.0):
    """Creates a state update function that replaces a state variable with signal value.

    Args:
        variable (str): The state variable name to update
        default_value (float, optional): Default value if not in signal. Defaults to 0.0.

    Returns:
        function: A state update function that sets variable to signal value
    """
    return lambda _1, _2, _3, state, signal: (
        variable,
        signal.get(variable, default_value),
    )


def add_suf(variable: str, default_value=0.0):
    """Creates a state update function that adds signal value to existing state value.

    Args:
        variable (str): The state variable name to update
        default_value (float, optional): Default value to add if variable not in signal. Defaults to 0.0.

    Returns:
        function: A state update function that adds signal value to the existing state value
    """
    return lambda _1, _2, _3, state, signal: (
        variable,
        signal.get(variable, default_value) + state[variable],
    )
