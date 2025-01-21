import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from typing import Callable  # type: ignore
import math  # type: ignore
from aztec_gddt.types import *


def target_mana_per_block(params: ModelParams) -> Mana:
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
    """

    # Derive drift and volatility
    drift = np.exp(-drift_decay_rate * curr_reward_time)
    volatility = volatility_coefficient * \
        np.exp(-0.5 * volatility_decay_rate * curr_reward_time)
    # Check the input values

    # Check initial conditions
    if prev_reward_time == 0 and prev_reward_value <= 0.0:
        raise ValueError(
            "Previous reward value must be greater than 0 for the first block"
        )
    # Check time consistency
    if prev_reward_time >= curr_reward_time:
        raise ValueError("Previous time must precede current time")

    # Compute the number of degrees of freedom
    degrees_of_freedom = (4 * drift_speed_adj * drift) / (volatility) ** 2

    # Check that the number of degrees of freedom is (essentially) an integer
    if not math.isclose(degrees_of_freedom, round(degrees_of_freedom)):
        raise ValueError("Degrees of freedom must be an integer")
    degrees_of_freedom = round(degrees_of_freedom)
    # Check that the number of degrees of freedom is greater than 2
    if degrees_of_freedom < 2:
        raise ValueError("Degrees of freedom must be at least 2")

    # STEPS:
    """The variable obtained by summing the squares of df (degrees of freedom) independent, standard normally distributed random variables:
        Y = (Z_1)^2 + (Z_2)^2 + ... + (Z_df)^2
        is chi-square distributed.
    """

    # Calculate the chi-square value
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
    k1 = (
        4 * drift_speed_adj *
        np.exp(-drift_speed_adj * (curr_reward_time - prev_reward_time))
    ) / (
        volatility**2 * (1 - np.exp(-drift_speed_adj *
                         (curr_reward_time - prev_reward_time)))
    )

    _lambda = prev_reward_value * k1

    k2 = (np.exp(-drift_speed_adj * (curr_reward_time - prev_reward_time))) / k1

    # Draw a random value from a standard normal distribution
    rnd_std_normal = rng.normal(0, 1)
    """
        Then, Compute x_chi_square
        $$
            x_chi_square = (rnd_std_normal + sqrt{_lambda})^2 +  y_chi_square (*)
        $$
    """
    x_chi_square = (rnd_std_normal + np.sqrt(_lambda)) ** 2 + y_chi_square

    """
        Thus, the final realization $r_t$ of $R_t$ given realization $r_s$ is $r_t := k_2*x_chi_square$.
    """

    reward = k2 * x_chi_square

    return reward


def raw_base_fee(
        # Parameters
        target_mana_per_block: Mana,

        # Oracle / Contract related
        l1_gas_price: WeiPerGas,
        l1_blobgas_price: WeiPerGas,
        juice_per_wei_price: JuicePerWei,
        proving_cost_per_mana_in_wei: WeiPerMana,
        congestion_multiplier: float,

        # Tx related
        blobs_per_block: int,
        l1_gas_per_block: Gas,
        l1_blobgas_per_block: Gas
) -> JuicePerMana:

    l1_gas_cost_in_wei_per_l2block: Wei = l1_gas_per_block * l1_gas_price
    l1_da_cost_in_wei_per_l2block: Wei = blobs_per_block * \
        l1_blobgas_per_block * l1_blobgas_price

    l1_cost_in_wei_per_l2block: Wei = l1_gas_cost_in_wei_per_l2block + \
        l1_da_cost_in_wei_per_l2block

    l1_cost_per_mana_in_wei: WeiPerMana = l1_cost_in_wei_per_l2block / target_mana_per_block

    wei_per_mana = l1_cost_per_mana_in_wei + proving_cost_per_mana_in_wei

    base_fee_in_wei_per_mana: WeiPerMana = wei_per_mana * congestion_multiplier

    base_fee_in_juice_per_mana: JuicePerMana = base_fee_in_wei_per_mana * juice_per_wei_price

    return base_fee_in_juice_per_mana


def excess_mana_fn(past_excess: Mana,
                   past_spent: Mana,
                   target_mana: Mana) -> Mana:
    if (past_excess + past_spent) > target_mana:
        return past_excess + past_spent - target_mana
    else:
        return 0


def proving_cost_fn(minimum_proving_cost_wei_per_mana: WeiPerMana,
                    proving_cost_modifier: float,
                    proving_cost_update_fraction: float) -> WeiPerMana:
    exp_term = math.exp(proving_cost_modifier / proving_cost_update_fraction)

    # TODO: include logic for bounding the update to a min-max range.

    return minimum_proving_cost_wei_per_mana * exp_term


def juice_per_wei_price_fn(minimum_fee_asset_per_wei: JuicePerWei,
                           fee_juice_price_modifier: float,
                           fee_asset_per_wei_update_fraction: float,
                           old_juice_per_wei_price: JuicePerWei,
                           max_fee_juice_price_relative_change: Percentage) -> JuicePerWei:

    exp_term = math.exp(fee_juice_price_modifier /
                        fee_asset_per_wei_update_fraction)
    new_juice_per_wei_price: JuicePerWei = minimum_fee_asset_per_wei * exp_term

    max_price: JuicePerWei = old_juice_per_wei_price * \
        (1 + max_fee_juice_price_relative_change)
    min_price: JuicePerWei = old_juice_per_wei_price * \
        (1 - max_fee_juice_price_relative_change)

    if new_juice_per_wei_price > max_price:
        return max_price
    elif new_juice_per_wei_price < min_price:
        return min_price
    else:
        return new_juice_per_wei_price


def compute_base_fee(params: ModelParams, state: ModelState) -> JuicePerMana:

    l1_gas_for_da: Gas = params['BLOBS_PER_BLOCK'] * \
        params['POINT_EVALUATION_PRECOMIPLE_GAS']

    l1_gas_per_block: Gas = params['L1_GAS_TO_PUBLISH'] + l1_gas_for_da + int(
        params['L1_GAS_TO_VERIFY'] / params['L2_SLOTS_PER_L2_EPOCH'])
    l1_blobgas_per_block: Gas = params['L1_BLOBGAS_PER_BLOB'] * \
        params['BLOBS_PER_BLOCK']
    juice_per_wei_price = state['oracle_price_juice_per_wei']
    proving_cost_per_mana_in_wei = state['oracle_proving_cost']
    congestion_multiplier = state['congestion_multiplier']

    return raw_base_fee(
        target_mana_per_block=target_mana_per_block(params),

        l1_gas_price=state['oracle_price_l1_gas'],
        l1_blobgas_price=state['oracle_price_l1_blobgas'],
        juice_per_wei_price=juice_per_wei_price,
        proving_cost_per_mana_in_wei=proving_cost_per_mana_in_wei,
        congestion_multiplier=congestion_multiplier,

        blobs_per_block=params['BLOBS_PER_BLOCK'],
        l1_gas_per_block=l1_gas_per_block,
        l1_blobgas_per_block=l1_blobgas_per_block)


def l2_block_cost_for_sequencer(params: ModelParams, state: ModelState) -> Juice:

    l1_gas_for_da: Gas = params['BLOBS_PER_BLOCK'] * \
        params['POINT_EVALUATION_PRECOMIPLE_GAS']

    l1_gas_per_block: Gas = params['L1_GAS_TO_PUBLISH'] + l1_gas_for_da + int(
        params['L1_GAS_TO_VERIFY'] / params['L2_SLOTS_PER_L2_EPOCH'])
    l1_blobgas_per_block: Gas = params['L1_BLOBGAS_PER_BLOB'] * \
        params['BLOBS_PER_BLOCK']
    juice_per_wei_price = state['market_price_juice_per_wei']
    proving_cost_per_mana_in_wei = 0.0
    congestion_multiplier = 1.0

    return raw_base_fee(
        target_mana_per_block=1,

        l1_gas_price=state['market_price_l1_gas'],
        l1_blobgas_price=state['market_price_l1_blobgas'],
        juice_per_wei_price=juice_per_wei_price,
        proving_cost_per_mana_in_wei=proving_cost_per_mana_in_wei,
        congestion_multiplier=congestion_multiplier,

        blobs_per_block=params['BLOBS_PER_BLOCK'],
        l1_gas_per_block=l1_gas_per_block,
        l1_blobgas_per_block=l1_blobgas_per_block)  # type: ignore


def expected_profit_per_tx(params: ModelParams,
                           state: ModelState,
                           max_fee,
                           prover_quote: Percentage,
                           tx_count: int) -> Juice:
    expected_cost_per_tx = l2_block_cost_for_sequencer(
        params, state) / tx_count # type: ignore
    expected_revenue_per_tx = max_fee * (1 - prover_quote)
    return expected_revenue_per_tx - expected_cost_per_tx
