from aztec_gddt.types import *
from random import random


DEFAULT_BOND_SIZE: Juice = 1

N_AGENTS = 300
DEFAULT_INITIAL_AGENTS = []
for i in range(N_AGENTS):
    a = Agent(uuid=str(i),
              commitment_bond=5 * random() * DEFAULT_BOND_SIZE,
              score=random()
              )
    DEFAULT_INITIAL_AGENTS.append(a)


invalid_epoch = Epoch(init_time_in_l1=-999,
                      validators=[],
                      slots=[Slot(-999, '', -999, -999, -999)],
                      time_until_E_EPOCH_QUOTE_ACCEPT=-999,
                      time_until_E_EPOCH_FINISH=-999)

DEFAULT_INITIAL_EPOCH = invalid_epoch
DEFAULT_LAST_EPOCH = invalid_epoch


DEFAULT_INITIAL_STATE = ModelState(
    l1_blocks_passed=0,
    delta_l1_blocks=0,
    l2_blocks_passed=0,
    agents=DEFAULT_INITIAL_AGENTS,
    validator_set=set(),
    PROVING_COST_MODIFIER=0.0,
    current_epoch=DEFAULT_INITIAL_EPOCH,
    last_epoch=DEFAULT_LAST_EPOCH,

    # Block Reward related values
    last_reward_time_in_l1=0,  # XXX
    last_reward=15_000,  # XXX

    # Market & Oracle Values
    market_price_juice_per_mana=5.0,  # TODO
    market_price_l1_gas=30,  # TODO
    market_price_l1_blobgas=5,  # TODO

    oracle_price_juice_per_mana=4.0,  # TODO
    oracle_price_l1_gas=28,  # TODO
    oracle_price_l1_blobgas=4,  # TODO

    update_time_oracle_price_juice_per_mana=-999,
    update_time_oracle_price_l1_gas=-999,
    update_time_oracle_price_l1_blobgas=-999,

    oracle_proving_cost=0,  # TODO
    congestion_multiplier=1.0,  # TODO
    excess_mana=0,  # TODO


    # State Metrics
    base_fee=10,  # TODO
    cumm_empty_blocks=0,
    cumm_unproven_epochs=0,
    cumm_dropped_tx=0,
    cumm_excl_tx=0,
    cumm_total_tx=0,
    cumm_resolved_epochs=0,
    cumm_finalized_epochs=0,
    cumm_mana_used_on_finalized_blocks=0,
    cumm_finalized_blocks=0
)

DEFAULT_PARAMS = ModelParams(label='default',
                             timestep_in_l1_blocks=1,

                             ### General ###
                             OVERHEAD_MANA_PER_TX=200_000,
                             MAXIMUM_MANA_PER_BLOCK=20_000_000,  # 20M or 40M
                             TST_TOTAL_SUPPLY=500_000_000,  # XXX
                             LAUNCH_VALUATION=1_500_000_000,  # XXX
                             L2_SLOTS_PER_L2_EPOCH=32,
                             L1_SLOTS_PER_L2_SLOT=3,
                             PROVER_SEARCH_PERIOD=13,
                             MIN_ORACLE_UPDATE_LAG_C=5,

                             ### Fee ###
                             RELATIVE_TARGET_MANA_PER_BLOCK=0.5,
                             BLOBS_PER_BLOCK=3,  # fixed

                             L1_GAS_TO_VERIFY=1_000_000,  # fixed
                             L2_SLOTS_PER_EPOCH=32,  # fixed
                             L1_GAS_TO_PUBLISH=150_000,  # fixed
                             L1_BLOBGAS_PER_BLOB=int(2 ** 17),  # fixed, ~131k
                             POINT_EVALUATION_PRECOMIPLE_GAS=50_000,

                             MINIMUM_MULTIPLIER_CONGESTION=1.0,  # fixed
                             MINIMUM_PROVING_COST=1.0,  # fixed
                             MINIMUM_FEE_JUICE_PER_WEI=1.0,  # fixed

                             UPDATE_FRACTION_CONGESTION=2_000_000,  # TODO
                             UPDATE_FRACTION_PROVING_COST=1.0,  # TODO
                             UPDATE_FRACTION_FEE_JUICE_PER_WEI=1.0,  # TODO

                             MAX_RELATIVE_CHANGE_CONGESTION=0.03,  # TODO
                             MAXIMUM_UPDATE_PERCENTAGE_C=0.03,

                             MAX_FEE_INFLATION_PER_BLOCK=0.10,  # TODO

                             ### Reward ###
                             BLOCK_REWARD_VOLATILITY=3.17e-6,
                             BLOCK_REWARD_DRIFT_DECAY_RATE=1e-2,
                             BLOCK_REWARD_SPEED_ADJ=5.5e-4,

                             ### Staking ###
                             MIN_STAKE=32,  # fixed
                             VALIDATOR_COMMITTEE_SIZE=128,  # min 128, likely around 300
                             SIGNATURES_NEEDED=0.5,

                             ### Slashing ###
                             BOND_SIZE=DEFAULT_BOND_SIZE,  # control, sweep
                             BOND_SLASH_PERCENT=1.0,  # control, fixed

                             ### Behavioural ###
                             AVERAGE_TX_COUNT_PER_SLOT=360,
                             PROVING_COST_MODIFICATION_E=0.1,
                             FEE_JUICE_PRICE_MODIFICATION_E=0.1,
                             ORACLE_UPDATE_FREQUENCY_E=0.5,
                             JUICE_PER_MANA_MEAN=5.0,
                             JUICE_PER_MANA_STD=2.0,
                             
                             # Misc
                             PROVING_COST_MODIFIER_INITIAL_C=5.0,
                             FEE_JUICE_PRICE_MODIFIER_INITIAL_C=5.0,
                             )
