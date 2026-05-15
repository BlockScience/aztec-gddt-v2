"""
Economic and fee mechanism functions for the Aztec GDDT v2 simulation.

This module implements the core economic mechanisms of the Aztec system including:
1. Block reward calculation with randomized decay
2. Fee calculation based on L1 costs, proving costs, and congestion
3. Base fee computation incorporating oracle prices and market conditions
4. Expected profit calculations for sequencers and provers

These functions represent the key economic dynamics of the Aztec rollup system and are used by the policy and state update functions in the logic.py module.
"""

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from typing import Callable  # type: ignore
import math  # type: ignore
from aztec_gddt.types import *


def target_mana_per_block(params: ModelParams) -> Mana:
    """
    Calculate the target mana usage per block.

    This function computes the target mana per block as a fraction of the maximum mana per block. The fraction is determined by the RELATIVE_TARGET_MANA_PER_BLOCK parameter.

    Args:
        params (ModelParams): Model parameters

    Returns:
        Mana: Target mana per block
    """
    return int(params['MAXIMUM_MANA_PER_BLOCK'] * params['RELATIVE_TARGET_MANA_PER_BLOCK'])


def block_reward(
    curr_reward_time: int,
    prev_reward_time: int,
    prev_reward_value: Token,
    drift_speed_adj: float,
    drift_decay_rate: float,
    volatility_coefficient: float,
    volatility_decay_rate: float
) -> Token:
    """
    High-level goal:
    Calculate the reward for a block based on the previous reward r_s and the time difference between current time t and previous time s.

    This implements a stochastic block reward model with the following properties:
    1. Expected value follows a deterministic decay curve over time
    2. Volatility is controlled and also decays over time
    3. Reward distribution has chi-square properties

    Args:
        curr_reward_time (int): Current block time (t)
        prev_reward_time (int): Previous reward time (s)
        prev_reward_value (Token): Previous reward value (r_s)
        drift_speed_adj (float): Parameter controlling the speed of drift
        drift_decay_rate (float): Parameter controlling decay rate of drift
        volatility_coefficient (float): Parameter controlling volatility magnitude
        volatility_decay_rate (float): Parameter controlling decay of volatility

    Returns:
        Token: New block reward value (r_t)

    Raises:
        ValueError: If input parameters don't satisfy model constraints
    """

    # Calculate drift and volatility parameters based on time
    drift = np.exp(-drift_decay_rate * curr_reward_time)
    volatility = volatility_coefficient * \
        np.exp(-0.5 * volatility_decay_rate * curr_reward_time)

    # Validate input parameters
    # Check initial conditions
    if prev_reward_time == 0 and prev_reward_value <= 0.0:
        raise ValueError(
            "Previous reward value must be greater than 0 for the first block"
        )
    # Check time consistency
    if prev_reward_time >= curr_reward_time:
        raise ValueError("Previous time must precede current time")

    # Compute the number of degrees of freedom for the chi-square distribution
    degrees_of_freedom = (4 * drift_speed_adj * drift) / (volatility) ** 2

    # Validate degrees of freedom is (approximately) an integer
    if not math.isclose(degrees_of_freedom, round(degrees_of_freedom)):
        raise ValueError("Degrees of freedom must be an integer")
    degrees_of_freedom = round(degrees_of_freedom)

    # Ensure degrees of freedom is at least 2 (required for the model)
    if degrees_of_freedom < 2:
        raise ValueError("Degrees of freedom must be at least 2")

    # STEPS:
    """The variable obtained by summing the squares of df (degrees of freedom) independent, standard normally distributed random variables:
        Y = (Z_1)^2 + (Z_2)^2 + ... + (Z_df)^2
        is chi-square distributed.
    """

    # Generate a random chi-square value with (degrees_of_freedom - 1) degrees of freedom
    rng = np.random.default_rng()  # default random variable generator
    y_chi_square = rng.chisquare(degrees_of_freedom - 1)

    """
    Compute lambda
    $$
        _lambda := r_s k_1(s,t)
    $$
        Note:
    $$
        k_1(s,t) := {4*speed_adj*e^{-speed_adj*(t-s)}} divided by {volatility^2*( 1 - e^{-speed_adj*(t-s)})},
        k_2(s,t) := {e^{-speed_adj(curr_reward_time-prev_reward_time)}} divided by {k_1(s,t)}
    $$
    """
    # Calculate k1 - a parameter used in the reward distribution
    k1 = (
        4 * drift_speed_adj *
        np.exp(-drift_speed_adj * (curr_reward_time - prev_reward_time))
    ) / (
        volatility**2 * (1 - np.exp(-drift_speed_adj *
                         (curr_reward_time - prev_reward_time)))
    )

    # Calculate lambda parameter for the distribution
    _lambda = prev_reward_value * k1

    # Calculate k2 - another parameter used in the reward formula
    k2 = (np.exp(-drift_speed_adj * (curr_reward_time - prev_reward_time))) / k1

    # Draw a random value from a standard normal distribution
    rnd_std_normal = rng.normal(0, 1)
    """
        Then, Compute x_chi_square
        $$
            x_chi_square = (rnd_std_normal + sqrt{_lambda})^2 +  y_chi_square (*)
        $$
    """
    # Calculate x_chi_square using the formula above
    x_chi_square = (rnd_std_normal + np.sqrt(_lambda)) ** 2 + y_chi_square

    """
        Thus, the final realization $r_t$ of $R_t$ given realization $r_s$ is $r_t := k_2*x_chi_square$.
    """
    # Calculate the final reward value
    reward = k2 * x_chi_square

    return reward


def raw_base_fee(
        # Parameters
        target_mana_per_block: Mana,

        # Oracle / Contract related
        l1_gas_price: GweiPerGas,
        l1_blobgas_price: GweiPerGas,
        juice_per_gwei_price: JuicePerGwei,
        proving_cost_per_mana_in_gwei: GweiPerMana,
        congestion_multiplier: float,

        # Tx related
        blobs_per_block: int,
        l1_gas_per_block: Gas,
        l1_blobgas_per_block: Gas
) -> JuicePerMana:
    """
    Calculate the raw base fee per mana unit.

    This function computes the base fee for transactions based on:
    1. L1 gas costs for blocks
    2. L1 blob gas costs for data availability
    3. Proving costs
    4. Network congestion
    5. Exchange rates between Juice, Gwei, and other units

    Args:
        target_mana_per_block (Mana): Target mana per block
        l1_gas_price (GweiPerGas): Current L1 gas price in Gwei per gas unit
        l1_blobgas_price (GweiPerGas): Current L1 blob gas price in Gwei per blob gas unit
        juice_per_gwei_price (JuicePerGwei): Exchange rate between Juice and Gwei
        proving_cost_per_mana_in_gwei (GweiPerMana): Cost of proving per mana unit in Gwei
        congestion_multiplier (float): Multiplier based on network congestion
        blobs_per_block (int): Number of blobs per block
        l1_gas_per_block (Gas): L1 gas used per block
        l1_blobgas_per_block (Gas): L1 blob gas used per block

    Returns:
        JuicePerMana: Base fee in Juice per mana unit
    """
    # Calculate L1 gas cost in Gwei for the entire L2 block
    l1_gas_cost_in_gwei_per_l2block: Gwei = l1_gas_per_block * l1_gas_price

    # Calculate data availability (DA) cost in Gwei for the entire L2 block
    l1_da_cost_in_gwei_per_l2block: Gwei = l1_blobgas_per_block * l1_blobgas_price

    # Total L1 cost combines both gas and DA costs
    l1_cost_in_gwei_per_l2block: Gwei = l1_gas_cost_in_gwei_per_l2block + \
        l1_da_cost_in_gwei_per_l2block

    # Calculate L1 cost per mana unit by dividing total block cost by target mana
    l1_cost_per_mana_in_gwei: GweiPerMana = l1_cost_in_gwei_per_l2block / target_mana_per_block

    # Total gwei cost per mana combines L1 costs and proving costs
    gwei_per_mana = l1_cost_per_mana_in_gwei + proving_cost_per_mana_in_gwei

    # Apply congestion multiplier to get adjusted base fee in Gwei per mana
    base_fee_in_gwei_per_mana: GweiPerMana = gwei_per_mana * congestion_multiplier

    # Convert from Gwei to Juice using the current exchange rate
    base_fee_in_juice_per_mana: JuicePerMana = base_fee_in_gwei_per_mana * juice_per_gwei_price

    return base_fee_in_juice_per_mana


def excess_mana_fn(past_excess: Mana,
                   past_spent: Mana,
                   target_mana: Mana) -> Mana:
    """
    Calculate excess mana after a block.

    This function determines how much mana is in excess of the target, which is used to calculate congestion and adjust fees accordingly.

    Args:
        past_excess (Mana): Excess mana from previous block
        past_spent (Mana): Mana spent in the current block
        target_mana (Mana): Target mana per block

    Returns:
        Mana: New excess mana value (0 if below target)
    """
    # If total mana (excess + spent) exceeds target, record the excess
    if (past_excess + past_spent) > target_mana:
        return past_excess + past_spent - target_mana
    else:
        # If below target, there is no excess
        return 0


def proving_cost_fn(minimum_proving_cost_gwei_per_mana: GweiPerMana,
                    proving_cost_modifier: float,
                    proving_cost_update_fraction: float) -> GweiPerMana:
    """
    Calculate the oracle value for proving cost per mana in Gwei.

    This function computes the proving cost based on:
    1. A minimum proving cost
    2. An external proving cost modifier
    3. A parameter controlling the rate of change

    Args:
        minimum_proving_cost_gwei_per_mana (GweiPerMana): Minimum proving cost
        proving_cost_modifier (float): Modifier for proving cost
        proving_cost_update_fraction (float): Parameter controlling update rate

    Returns:
        GweiPerMana: Proving cost per mana in Gwei
    """
    # Calculate exponential term based on modifier and update fraction
    exp_term = math.exp(proving_cost_modifier / proving_cost_update_fraction)

    # Apply exponential term to minimum cost
    return minimum_proving_cost_gwei_per_mana * exp_term


def juice_per_gwei_price_fn(minimum_fee_asset_per_gwei: JuicePerGwei,
                            fee_juice_price_modifier: float,
                            fee_asset_per_gwei_update_fraction: float,
                            old_juice_per_gwei_price: JuicePerGwei,
                            max_fee_juice_price_relative_change: Percentage) -> JuicePerGwei:
    """
    Calculate the Juice per Gwei exchange rate.

    This function updates the Juice/Gwei exchange rate based on:
    1. A minimum exchange rate
    2. An external price modifier
    3. A parameter controlling update rate
    4. Bounds on the relative change

    Args:
        minimum_fee_asset_per_gwei (JuicePerGwei): Minimum exchange rate
        fee_juice_price_modifier (float): External modifier for price
        fee_asset_per_gwei_update_fraction (float): Parameter controlling update rate
        old_juice_per_gwei_price (JuicePerGwei): Previous exchange rate
        max_fee_juice_price_relative_change (Percentage): Maximum allowed relative change

    Returns:
        JuicePerGwei: Updated Juice per Gwei exchange rate
    """
    # Calculate new price using exponential update
    exp_term = math.exp(fee_juice_price_modifier /
                        fee_asset_per_gwei_update_fraction)
    new_juice_per_gwei_price: JuicePerGwei = minimum_fee_asset_per_gwei * exp_term

    # Calculate bounds on price change
    max_price: JuicePerGwei = old_juice_per_gwei_price * \
        (1 + max_fee_juice_price_relative_change)
    min_price: JuicePerGwei = old_juice_per_gwei_price * \
        (1 - max_fee_juice_price_relative_change)

    # Apply bounds: if price exceeds bounds, use bound value
    if new_juice_per_gwei_price > max_price:
        return max_price
    elif new_juice_per_gwei_price < min_price:
        return min_price
    else:
        return new_juice_per_gwei_price


def compute_base_fee(params: ModelParams, state: ModelState) -> JuicePerMana:
    """
    Compute the current base fee per mana unit.

    This function calculates the base fee for transactions by:
    1. Determining L1 costs for gas and data availability
    2. Incorporating oracle prices for L1 resources
    3. Adding proving costs
    4. Applying the congestion multiplier
    5. Converting to Juice units

    Args:
        params (ModelParams): Model parameters
        state (ModelState): Current model state

    Returns:
        JuicePerMana: Current base fee in Juice per mana unit
    """
    # Calculate L1 gas needed for data availability proofs
    l1_gas_for_da: Gas = params['BLOBS_PER_BLOCK'] * \
        params['POINT_EVALUATION_PRECOMIPLE_GAS']

    # Calculate total L1 gas per block (publish + DA + verification)
    # Note: verification gas is amortized across the entire epoch
    l1_gas_per_block: Gas = params['L1_GAS_TO_PUBLISH'] + l1_gas_for_da + int(
        params['L1_GAS_TO_VERIFY'] / params['L2_SLOTS_PER_L2_EPOCH'])

    # Calculate total blob gas per block
    l1_blobgas_per_block: Gas = params['L1_BLOBGAS_PER_BLOB'] * \
        params['BLOBS_PER_BLOCK']

    # Get current oracle prices and multipliers from state
    juice_per_gwei_price = state['oracle_price_juice_per_gwei']
    proving_cost_per_mana_in_gwei = state['oracle_proving_cost']
    congestion_multiplier = state['congestion_multiplier']

    # Call raw_base_fee with all required parameters
    return raw_base_fee(
        target_mana_per_block=target_mana_per_block(params),

        l1_gas_price=state['oracle_price_l1_gas'],
        l1_blobgas_price=state['oracle_price_l1_blobgas'],
        juice_per_gwei_price=juice_per_gwei_price,
        proving_cost_per_mana_in_gwei=proving_cost_per_mana_in_gwei,
        congestion_multiplier=congestion_multiplier,

        blobs_per_block=params['BLOBS_PER_BLOCK'],
        l1_gas_per_block=l1_gas_per_block,
        l1_blobgas_per_block=l1_blobgas_per_block)

def compute_base_fee_congestionless(params: ModelParams, state: ModelState):
    modified_state = state.copy()
    modified_state['congestion_multiplier'] = params['MINIMUM_MULTIPLIER_CONGESTION']
    return compute_base_fee(params, modified_state)


def compute_base_fee_proverless(params: ModelParams, state: ModelState):
    modified_state = state.copy()
    modified_state['oracle_proving_cost'] = 0
    return compute_base_fee(params, modified_state)

def l2_block_cost_for_sequencer(params: ModelParams, state: ModelState) -> Juice:
    """
    Calculate the cost a sequencer incurs to produce an L2 block.

    This function computes the cost to a sequencer for publishing an L2 block,
    based on L1 gas prices, blob data availability, and exchange rates. This is
    different from the base fee as it doesn't include proving costs or congestion.

    Args:
        params (ModelParams): Model parameters
        state (ModelState): Current model state

    Returns:
        Juice: Cost for the sequencer to produce a block, in Juice units
    """
    # Calculate L1 gas for data availability proofs
    l1_gas_for_da: Gas = params['BLOBS_PER_BLOCK'] * \
        params['POINT_EVALUATION_PRECOMIPLE_GAS']

    # Calculate total L1 gas for block publication (excluding verification)
    l1_gas_per_block: Gas = params['L1_GAS_TO_PUBLISH'] + l1_gas_for_da

    # Calculate blob gas needed for block
    l1_blobgas_per_block: Gas = params['L1_BLOBGAS_PER_BLOB'] * \
        params['BLOBS_PER_BLOCK']

    # Use market prices (not oracle prices) for sequencer calculations
    juice_per_gwei_price = state['market_price_juice_per_gwei']

    # Sequencers don't pay proving costs directly
    proving_cost_per_mana_in_gwei = 0.0

    # No congestion multiplier for sequencer costs
    congestion_multiplier = 1.0

    # Calculate raw cost using raw_base_fee function
    # Note: target_mana_per_block=1 because we want cost per block, not per mana
    return raw_base_fee(
        target_mana_per_block=1,

        l1_gas_price=state['market_price_l1_gas'],
        l1_blobgas_price=state['market_price_l1_blobgas'],
        juice_per_gwei_price=juice_per_gwei_price,
        proving_cost_per_mana_in_gwei=proving_cost_per_mana_in_gwei,
        congestion_multiplier=congestion_multiplier,

        blobs_per_block=params['BLOBS_PER_BLOCK'],
        l1_gas_per_block=l1_gas_per_block,
        l1_blobgas_per_block=l1_blobgas_per_block)  # type: ignore


def expected_profit_per_tx(params: ModelParams,
                           state: ModelState,
                           max_fee,
                           prover_quote: Percentage,
                           tx_count: int) -> Juice:
    """
    Calculate the expected profit per transaction for a sequencer.

    This function computes the profit a sequencer can expect from including
    a transaction, based on:
    1. The cost to produce the block
    2. The base fee for transactions
    3. The prover's quote (percentage of fees they take) 
    4. The number of transactions in the block

    Args:
        params (ModelParams): Model parameters
        state (ModelState): Current model state
        max_fee: Maximum fee user is willing to pay
        prover_quote (Percentage): Percentage of fees going to the prover
        tx_count (int): Number of transactions in the block

    Returns:
        Juice: Expected profit per transaction in Juice units
    """
    # Calculate cost per transaction by dividing block cost by transaction count
    expected_cost_per_tx = l2_block_cost_for_sequencer(
        params, state) / tx_count # type: ignore
    
    prover_factor = (state['base_fee'] - state['base_fee_proverless']) / state['base_fee']

    expected_revenue_per_tx = state['base_fee'] * (1 - prover_factor)

    # Profit is revenue minus cost
    return expected_revenue_per_tx - expected_cost_per_tx
