from aztec_gddt.types import *
from aztec_gddt.default_params import *
from aztec_gddt.structure import MODEL_BLOCKS
from aztec_gddt.utils import policy_aggregator
from copy import deepcopy
from cadCAD.configuration import Experiment  # type: ignore
from cadCAD.configuration.utils import config_sim  # type: ignore
from cadCAD.tools.preparation import sweep_cartesian_product # type: ignore
from random import sample


def scenario_group_1_experiment(N_timesteps: int = 100, 
                                N_samples: int = 2,
                                N_config_sample: float = 300) -> Experiment:

    control_params_to_sweep: dict = {
        'RELATIVE_TARGET_MANA_PER_BLOCK': [0.5, 0.9],
        'MAXIMUM_MANA_PER_BLOCK': [20_000_000, 40_000_000],
        'MINIMUM_MULTIPLIER_CONGESTION': [1_000_000, 4_000_000, 10_000_000],
        'RELATIVE_UPDATE_FRACTION_CONGESTION': [0.1, 1.0, 10.0],
        'OVERHEAD_MANA_PER_TX': [1_000, 10_000, 50_000],
        'MAXIMUM_UPDATE_PERCENTAGE_C': [0.01, 0.03]
    }

    env_params_to_sweep: dict = {
        'JUICE_PER_WEI_MEAN': [1.1e-15, 10e-15],
        'JUICE_PER_WEI_COV': [0.03, 0.30]
    }


    default_params = {k: [v] for k, v in DEFAULT_PARAMS.items()}

    params_to_sweep = {**default_params, **
                       env_params_to_sweep, **control_params_to_sweep}
    
    prepared_params = sweep_cartesian_product(params_to_sweep)

    states_list = [DEFAULT_INITIAL_STATE, DEFAULT_INITIAL_STATE]

    exp = Experiment()
    for state in states_list:
        simulation_parameters = {"N": N_samples, "T": range(N_timesteps), "M": prepared_params}
        sim_config = config_sim(simulation_parameters)  # type: ignore
        exp.append_configs(
            sim_configs=sim_config,
            initial_state=state,
            partial_state_update_blocks=MODEL_BLOCKS,
            policy_ops=[policy_aggregator],
        )

    if int(N_config_sample) > 0:
        exp.configs = sample(exp.configs, int(N_config_sample))

    return exp



# def scenario_group_2_experiment(N_timesteps: int = 1_000, 
#                                 N_samples: int = 2,
#                                 N_config_sample: float = 30) -> Experiment:

#     control_params_to_sweep: dict = {
#         'MAX_FEE_INFLATION_RELATIVE_MEAN': [0.0, 0.5, 1.0, 2.0],
#         'RELATIVE_TARGET_MANA_PER_BLOCK': [0.1, 0.5, 0.9],
#         'MAXIMUM_MANA_PER_BLOCK': [20_000_000, 40_000_000],
#         'MINIMUM_MULTIPLIER_CONGESTION': [1_000_000, 4_000_000],
#         'RELATIVE_UPDATE_FRACTION_CONGESTION': [0.5],
#         'OVERHEAD_MANA_PER_TX': [1_000, 10_000, 50_000],

#         'MAXIMUM_UPDATE_PERCENTAGE_C': [0.3]
#     }

#     env_params_to_sweep: dict = {
#         'JUICE_PER_WEI_MEAN': [1.1e-15, 10e-15],
#         'JUICE_PER_WEI_COV': [0.05, 0.30],
#         'PROVING_COST_INITIAL_IN_USD_PER_TX_C': [0.03, 0.10],
#         'FEE_JUICE_PRICE_MODIFIER_INITIAL_C': [1.0],
#     }


#     default_params = {k: [v] for k, v in DEFAULT_PARAMS.items()}

#     params_to_sweep = {**default_params, **
#                        env_params_to_sweep, **control_params_to_sweep}
    
#     prepared_params = sweep_cartesian_product(params_to_sweep)

#     states_list = [DEFAULT_INITIAL_STATE, DEFAULT_INITIAL_STATE]

#     exp = Experiment()
#     for state in states_list:
#         simulation_parameters = {"N": N_samples, "T": range(N_timesteps), "M": prepared_params}
#         sim_config = config_sim(simulation_parameters)  # type: ignore
#         exp.append_configs(
#             sim_configs=sim_config,
#             initial_state=state,
#             partial_state_update_blocks=MODEL_BLOCKS,
#             policy_ops=[policy_aggregator],
#         )

#     if int(N_config_sample) > 0:
#         exp.configs = sample(exp.configs, int(N_config_sample))

#     return exp