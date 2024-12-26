import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from typing import Callable # type: ignore
import math # type: ignore



def block_reward(
    curr_reward_time: int,
    prev_reward_time: int,
    prev_reward_value: float,
    drift_speed_adj: float = 5.5e-4,
    drift_decay_rate: float = 3.17e-6,
    volatility_coefficient: float = 3.17e-6,
    volatility_decay_rate: float = 1e-2
) -> float:
    """
    High-level goal:
    Calculate the reward for a block based on the previous reward r_s and the time difference between current time t and previous time s.
    """

    
    # Derive drift and volatility
    drift = np.exp(-drift_decay_rate * curr_reward_time)
    volatility = volatility_coefficient * np.exp(-0.5 * volatility_decay_rate * curr_reward_time)
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
        4 * drift_speed_adj * np.exp(-drift_speed_adj * (curr_reward_time - prev_reward_time))
    ) / (
        volatility**2 * (1 - np.exp(-drift_speed_adj * (curr_reward_time - prev_reward_time)))
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