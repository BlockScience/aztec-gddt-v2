from aztec_gddt.types import *
from random import random

DEFAULT_N_TIMESTEPS = 1_000
DEFAULT_BOND_SIZE: Juice = 1

N_AGENTS = 512
DEFAULT_INITIAL_AGENTS = {}

AGGREGATE_PROVER = Agent(uuid='prover',
                         stake=1000 * DEFAULT_BOND_SIZE,
                         score=float('nan'))

DEFAULT_INITIAL_AGENTS[AGGREGATE_PROVER.uuid] = AGGREGATE_PROVER

for i in range(N_AGENTS):
    a = Agent(uuid=str(i),
              stake=5 * random() * DEFAULT_BOND_SIZE,
              score=random()
              )
    DEFAULT_INITIAL_AGENTS[a.uuid] = a


invalid_epoch = Epoch(init_time_in_l1=-999,
                      validators=[],
                      slots=[Slot(-999, '', -999, -999, -999)],
                      time_until_E_EPOCH_QUOTE_ACCEPT=-999,
                      time_until_E_EPOCH_FINISH=-999)

DEFAULT_INITIAL_EPOCH = invalid_epoch
DEFAULT_LAST_EPOCH = invalid_epoch

DEFAULT_INITIAL_STATE = ModelState(
    timestep=0,
    l1_blocks_passed=0,
    delta_l1_blocks=0,
    l2_blocks_passed=0,
    agents=DEFAULT_INITIAL_AGENTS,
    validator_set=set(),
    current_epoch=DEFAULT_INITIAL_EPOCH,
    last_epoch=DEFAULT_LAST_EPOCH,

    # Block Reward related values
    last_reward_time_in_l1=0,  # XXX
    last_reward=1_500,  # Roughly based on the 1st month reward

    # Market & Oracle Values
    market_price_juice_per_gwei=float('nan'),
    market_price_l1_gas=float('nan'),  # TODO
    market_price_l1_blobgas=float('nan'),  # TODO

    oracle_price_juice_per_gwei=float('nan'),
    oracle_price_l1_gas=float('nan'),  # TODO
    oracle_price_l1_blobgas=float('nan'),  # TODO

    update_time_oracle_price_juice_per_gwei=-999,
    update_time_oracle_price_l1_gas=-999,
    update_time_oracle_price_l1_blobgas=-999,

    oracle_proving_cost=float('nan'),  # TODO
    congestion_multiplier=float('nan'),  # TODO
    excess_mana=0,  # TODO


    # State Metrics
    base_fee=float('nan'),  # TODO
    cumm_empty_blocks=0,
    cumm_unproven_epochs=0,
    cumm_dropped_tx=0,
    cumm_excl_tx=0,
    cumm_total_tx=0,
    cumm_resolved_epochs=0,
    cumm_finalized_epochs=0,
    cumm_mana_used_on_finalized_blocks=0,
    cumm_finalized_blocks=0,
    cumm_blocks_with_collected_signatures=0,
    cumm_blocks_with_enough_signatures=0,
    slash_count=0,
    slash_amount=0.0
)

DEFAULT_PARAMS = ModelParams(label='default',
                             timestep_in_l1_blocks=1,
                             N_timesteps=DEFAULT_N_TIMESTEPS,

                             ### General ###
                             OVERHEAD_MANA_PER_TX=45_000,
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
                             L1_GAS_TO_PUBLISH=150_000,  # fixeds
                             L1_BLOBGAS_PER_BLOB=int(2 ** 17),  # fixed, ~131k
                             POINT_EVALUATION_PRECOMIPLE_GAS=50_000,  # fixed
                             MINIMUM_MULTIPLIER_CONGESTION=1_000_000,  # sweep
                             MAXIMUM_MULTIPLIER_CONGESTION_RATIO=10, # XXX, roughly equivalent to a max increase of 22026x
                             RELATIVE_UPDATE_FRACTION_CONGESTION=0.5,  # sweep
                             MAX_RELATIVE_CHANGE_CONGESTION=0.03,  # TODO
                             MAXIMUM_UPDATE_PERCENTAGE_C=0.03,
                             PROVING_COST_INITIAL_IN_USD_PER_TX_C=0.03,

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
                             PROVING_COST_MODIFICATION_E=0.0,
                             FEE_JUICE_PRICE_MODIFICATION_E=0.0,
                             ORACLE_UPDATE_FREQUENCY_E=0.5,
                             JUICE_PER_GWEI_MEAN=1.1e-6,
                             JUICE_PER_GWEI_COV=0.10,
                             INITIAL_INFLATION_ESTIMATE=0.0,
                             MAX_FEE_INFLATION_RELATIVE_MEAN=1.0,
                             MAX_FEE_INFLATION_RELATIVE_STD=0.1,

                             GWEI_PER_L1GAS_MEAN=30,
                             GWEI_PER_L1GAS_COV=0.01,
                             GWEI_PER_L1BLOBGAS_MEAN=5,
                             GWEI_PER_L1BLOBGAS_COV=0.05,
                             SEQUENCER_L1_GAS_PRICE_THRESHOLD_E=float('nan'),
                             TOTAL_MANA_MULTIPLIER_E=1.0,
                             SIGNATURE_SKIP_PROBABILITY=0.0,
                             MAX_VALIDATORS_TO_SLASH=0.0,

                             PROVER_QUOTE_LOWER_BOUND=0.1,
                             PROVER_QUOTE_RANGE=0.8,
                             PROVER_QUOTE_MODE=0.3,

                             JUICE_PER_GWEI_SCENARIO=JuiceGweiExchangeRateScenario.Stochastic,

                             # Exogenous
                             market_price_eth=3300,
                             )
