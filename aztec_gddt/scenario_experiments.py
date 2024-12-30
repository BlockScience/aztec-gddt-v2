from aztec_gddt.types import *
from aztec_gddt.default_params import *
from copy import deepcopy


def scenario_group_1_experiment() -> None:

    params_to_sweep: dict = {
        'general': {
            'OVERHEAD_MANA_PER_TX': [1_000, 10_000, 20_000, 50_000],
            'MAXIMUM_MANA_PER_BLOCK': [20_000_000, 40_000_000]
        },
        'fee': {
            'RELATIVE_TARGET_MANA_PER_BLOCK': [0.1, 0.5, 0.9],
            'MINIMUM_MULTIPLIER_CONGESTION': [1.0],
            'MAX_RELATIVE_CHANGE_CONGESTION': [0.03],
        },
        'behavior': {
            'PROVING_COST_MODIFICATION_E': [0.0, 0.1, -0.1],
            'FEE_JUICE_PRICE_MODIFICATION_E': [0.0, 0.1, -0.1],
            'ORACLE_UPDATE_FREQUENCY_E': [0.9, 0.1]
        }
    }

    initial_state_to_sweep: dict = {
        'oracle_proving_cost': [0.0],  # TODO, PROVING_COST_MODIFIER_INITIAL_C
        # TODO, FEE_JUICE_PRICE_MODIFIER_INITIAL_C
        'oracle_price_juice_per_mana': [0.0]
    }







    initial_state = deepcopy(DEFAULT_INITIAL_STATE)
    params = deepcopy(DEFAULT_PARAMS)
