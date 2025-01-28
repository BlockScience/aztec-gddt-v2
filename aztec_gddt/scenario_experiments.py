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
        'JUICE_PER_GWEI_MEAN': [1.1e-15, 10e-15],
        'JUICE_PER_GWEI_COV': [0.03, 0.30]
    },
    N_timesteps=3_000,
    N_samples=5,
    N_config_sample=-1,
    relevant_per_trajectory_metrics=['T-M1', 'T-M2', 'T-M3'],
    relevant_per_trajectory_group_metrics=['TG-M1', 'TG-M2', 'TG-M3'],
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
    N_config_sample=-1,
    relevant_per_trajectory_metrics=['T-M4'],
    relevant_per_trajectory_group_metrics=['TG-M4', 'TG-M5'],
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
        'GWEI_PER_L1GAS_MEAN': [30, 100],
        'AVERAGE_TX_COUNT_PER_SLOT': [360, 1080]
    },
    N_timesteps=3_000,
    N_samples=5,
    N_config_sample=-1,
    relevant_per_trajectory_metrics=['T-M5', 'T-M6'],
    relevant_per_trajectory_group_metrics=['TG-M6', 'TG-M7'],
)


experiment_feemech_oracle_sensitivity = ExperimentParamSpec(
    params_swept_control={
        'MAXIMUM_UPDATE_PERCENTAGE_C': [0.01, 0.03],
    },
    params_swept_env={
        'ORACLE_UPDATE_PRICE_FREQUENCY': [0.10, 0.50],
        'PROVING_COST_MODIFICATION_E': [-0.001, 0.0, 0.001],  # TODO
        'FEE_JUICE_PRICE_MODIFICATION_E': [-0.001, 0.0, 0.001],  # TODO
    },
    N_timesteps=3_000,
    N_samples=100,
    N_config_sample=-1,
    relevant_per_trajectory_metrics=['T-M7a', 'T-M7b', 'T-M8'],
    relevant_per_trajectory_group_metrics=['TG-M8a', 'TG-M8b', 'TG-M9'],
)

experiment_stakeslash_resume_inactivity = ExperimentParamSpec(
    params_swept_control={
        'PERCENTAGE_STAKE_SLASHED_C': [0.00, 0.10, 1.00],
        'VALIDATOR_COMMITTEE_SIZE': [128, 256, 512],
        'SIGNATURES_NEEDED': [0.33, 0.50, 0.66]
    },
    params_swept_env={
        'SIGNATURE_SKIP_PROBABILITY': [0.00, 0.02]
    },
    N_timesteps=3_000,
    N_samples=50,
    N_config_sample=-1,
    relevant_per_trajectory_metrics=['T-M9'],
    relevant_per_trajectory_group_metrics=['TG-M10'],
)


experiment_stakeslash_validator_eject = ExperimentParamSpec(
    params_swept_control={
        'PERCENTAGE_STAKE_SLASHED_C': [0.00, 0.10, 1.00],
        'VALIDATOR_COMMITTEE_SIZE': [128, 256, 512],
        'SIGNATURES_NEEDED': [0.33, 0.50, 0.66]
    },
    params_swept_env={
        'MAX_VALIDATORS_TO_SLASH': [0.0, 1.0]
    },
    N_timesteps=3_000,
    N_samples=50,
    N_config_sample=-1,
    relevant_per_trajectory_metrics=['T-M10'],
    relevant_per_trajectory_group_metrics=['TG-M11'],
)

experiment_l2_congestion = ExperimentParamSpec(
    params_swept_control={
        'RELATIVE_TARGET_MANA_PER_BLOCK': [0.50, 0.90],
        'MAXIMUM_MANA_PER_BLOCK': [20_000_000, 40_000_000],
    },
    params_swept_env={
        'SEQUENCER_L1_GAS_PRICE_THRESHOLD_E': [100, 1_000],
        'TOTAL_MANA_MULTIPLIER_E': [1.0, 10.0]
    },
    N_timesteps=3_000,
    N_samples=50,
    N_config_sample=-1,
    relevant_per_trajectory_metrics=['T-M11'],
    relevant_per_trajectory_group_metrics=['TG-M12', 'TG-M13'],
)
