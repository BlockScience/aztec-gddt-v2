from aztec_gddt.helper_types import ExperimentParamSpec


### Classes ###

# Scenario Specs


experiment_feemech_volatility = ExperimentParamSpec(
    params_swept_control={
        'RELATIVE_TARGET_MANA_PER_BLOCK': [0.5, 0.9],
        'MAXIMUM_MANA_PER_BLOCK': [20_000_000, 40_000_000],
        'MINIMUM_MULTIPLIER_CONGESTION': [1_000_000, 4_000_000, 10_000_000],
        'RELATIVE_UPDATE_FRACTION_CONGESTION': [0.1, 1.0, 10.0],
        'OVERHEAD_MANA_PER_TX': [1_000, 10_000, 50_000],
        'MAXIMUM_UPDATE_PERCENTAGE_C': [0.01, 0.03]
    },
    params_swept_env={
        'JUICE_PER_WEI_MEAN': [1.1e-15, 10e-15],
        'JUICE_PER_WEI_COV': [0.03, 0.30]
    },
    N_timesteps=3_000,
    N_samples=5,
    N_config_sample=-1
)

experiment_feemech_l2_cost_censorship = ExperimentParamSpec(
    params_swept_control={
        'MAX_FEE_INFLATION_RELATIVE_STD': [0.05, 0.1, 0.2],
        'RELATIVE_TARGET_MANA_PER_BLOCK': [0.5, 0.9],
        'MAXIMUM_MANA_PER_BLOCK': [20_000_000, 40_000_000],
        'MINIMUM_MULTIPLIER_CONGESTION': [1_000_000, 4_000_000, 10_000_000],
        'RELATIVE_UPDATE_FRACTION_CONGESTION': [0.1, 1.0, 10.0],
        'OVERHEAD_MANA_PER_TX': [1_000, 10_000, 50_000],
        'MAXIMUM_UPDATE_PERCENTAGE_C': [0.01, 0.03]
    },
    params_swept_env={
        'MAX_FEE_INFLATION_RELATIVE_MEAN': [0.5, 1.0, 1.5],
        'MAX_FEE_INFLATION_RELATIVE_STD': [0.02, 0.10, 0.50]
    },
    N_timesteps=3_000,
    N_samples=3,
    N_config_sample=-1
)

experiment_feemech_shock_analysis = ExperimentParamSpec(
    params_swept_control={
        'MAX_FEE_INFLATION_RELATIVE_STD': [0.05, 0.1, 0.2],
        'RELATIVE_TARGET_MANA_PER_BLOCK': [0.5, 0.9],
        'MAXIMUM_MANA_PER_BLOCK': [20_000_000, 40_000_000],
        'MINIMUM_MULTIPLIER_CONGESTION': [1_000_000, 4_000_000, 10_000_000],
        'RELATIVE_UPDATE_FRACTION_CONGESTION': [0.1, 1.0, 10.0],
        'OVERHEAD_MANA_PER_TX': [1_000, 10_000, 50_000],
        'MAXIMUM_UPDATE_PERCENTAGE_C': [0.01, 0.03]
    },
    params_swept_env={
        # TODO
    },
    N_timesteps=3_000,
    N_samples=5,
    N_config_sample=-1
)


experiment_feemech_oracle_sensitivity = ExperimentParamSpec(
    params_swept_control={
        'MAXIMUM_UPDATE_PERCENTAGE_C': [0.01, 0.03],
    },
    params_swept_env={
        'ORACLE_UPDATE_PRICE_FREQUENCY': [0.10, 0.50],
        'PROVING_COST_MODIFICATION_E': ["-", "=", "+"], # TODO
        'FEE_JUICE_PRICE_MODIFICATION_E': ["-", "=", "+"], # TODO
    },
    N_timesteps=3_000,
    N_samples=100,
    N_config_sample=-1
)

experiment_stakeslash_resume_inactivity = ExperimentParamSpec(
    params_swept_control={
        'PERCENTAGE_STAKE_SLASHED_C': [0.00, 0.10, 1.00],
        'VALIDATOR_COMMITTEE_SIZE_C': [128, 256, 512],
        'SIGNATURED_NEEDED_PERCENTAGE_C': [0.51, 0.98]
    },
    params_swept_env={
        'PROBABILITY_SLASHABLE_ACTION_E': [0.00_1, 0.01_0, 0.10_0]
    },
    N_timesteps=3_000,
    N_samples=50,
    N_config_sample=-1
)


experiment_stakeslash_validator_eject = ExperimentParamSpec(
    params_swept_control={
        'PERCENTAGE_STAKE_SLASHED_C': [0.00, 0.10, 1.00],
        'VALIDATOR_COMMITTEE_SIZE_C': [128, 256, 512],
        'SIGNATURED_NEEDED_PERCENTAGE_C': [0.51, 0.98]
    },
    params_swept_env={
        'PROBABILITY_SLASHABLE_ACTION_E': [0.00_1, 0.01_0, 0.10_0]
    },
    N_timesteps=3_000,
    N_samples=50,
    N_config_sample=-1
)

experiment_l2_congestion = ExperimentParamSpec(
    params_swept_control={
        'RELATIVE_TARGET_MANA_PER_BLOCK': [0.50, 0.90],
        'MAXIMUM_MANA_PER_BLOCK': [20_000_000, 40_000_000],
    },
    params_swept_env={
        'SEQUENCER_L1_GAS_PRICE_THRESHOLD_E': ['low', 'high'],
        'USER_MAXIMUM_MANA_E': ['low', 'high']
    },
    N_timesteps=3_000,
    N_samples=50,
    N_config_sample=-1
)