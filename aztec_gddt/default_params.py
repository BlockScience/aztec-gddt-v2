from aztec_gddt.types import *
from random import random

# Suffix conventions:
# _E: Exogenous parameter (determined by external factors)
# _C: Control parameter (set by protocol / governance)

# Number of timesteps in the simulation
DEFAULT_N_TIMESTEPS = 1_000

# Minimum ETH stake required to become a validator
DEFAULT_MIN_STAKE: ETH = 32

# Total number of agents in the simulation
N_AGENTS = 512
# Dictionary to store all initial agents
DEFAULT_INITIAL_AGENTS = {}

# Aggregate prover agent
AGGREGATE_PROVER = Agent(uuid='prover',
                         stake=float('inf'),
                         score=float('nan'))

DEFAULT_INITIAL_AGENTS[AGGREGATE_PROVER.uuid] = AGGREGATE_PROVER

# Initialize random agents with varying stake and score values
for i in range(N_AGENTS):
    a = Agent(uuid=str(i),
              stake=5 * random() * DEFAULT_MIN_STAKE,
              score=random()
              )
    DEFAULT_INITIAL_AGENTS[a.uuid] = a


invalid_epoch = Epoch(init_time_in_l1=-999,
                      validators=[],
                      slots=[Slot(-999, '', -999, -999, -999)],
                      time_until_E_EPOCH_FINISH=-999)

DEFAULT_INITIAL_EPOCH = invalid_epoch
DEFAULT_LAST_EPOCH = invalid_epoch

# Initial state of the model
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
    # L1 block number when the last reward was distributed
    last_reward_time_in_l1=0,
    last_reward_to_provers=0,
    last_reward=1_500,  # Roughly based on the 1st month reward
    last_reward_per_prover=0.0,

    # Market & Oracle Values
    # Market price of exchange rate and L1 (blob) gas in Gwei
    market_price_juice_per_gwei=float('nan'),
    market_price_l1_gas=float('nan'),
    market_price_l1_blobgas=float('nan'), 
    # Oracle reported values for exchange rate and L1 (blob) gas in Gwei
    oracle_price_juice_per_gwei=float('nan'),
    oracle_price_l1_gas=float('nan'),
    oracle_price_l1_blobgas=float('nan'),

    update_time_oracle_price_juice_per_gwei=-999,
    update_time_oracle_price_l1_gas=-999,
    update_time_oracle_price_l1_blobgas=-999,

    oracle_proving_cost=float('nan'), 
    congestion_multiplier=float('nan'), 
    excess_mana=0, 


    # State Metrics
    base_fee=float('nan'), 
    base_fee_congestionless=float('nan'),
    base_fee_proverless=float('nan'),
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
    cumm_fee_paid=0.0,
    cumm_fee_burnt=0.0,
    cumm_fee_to_provers=0.0,
    cumm_fee_to_sequencers=0.0,
    slash_count=0,
    slash_amount=0.0
)

# Default model parameters for the simulation
DEFAULT_PARAMS = ModelParams(label='default',
                             timestep_in_l1_blocks=1,
                             N_timesteps=DEFAULT_N_TIMESTEPS,

                             ### General ###
                             OVERHEAD_MANA_PER_TX=45_000,
                             MAXIMUM_MANA_PER_BLOCK=20_000_000,  # 20M or 40M
                             # Initial simulation exchange rate is initialized through supply and valuation assumptions
                             # Current assumptions set by BSci for sim to run, need to be adjusted by anyone running the sim for concrete assumptions
                             SIM_TST_TOTAL_SUPPLY=100_000_000,
                             SIM_LAUNCH_VALUATION=100_000_000,
                             L2_SLOTS_PER_L2_EPOCH=32,
                             L1_SLOTS_PER_L2_SLOT=3,
                             # Oracle Lifetime value
                             MIN_ORACLE_UPDATE_LAG_C=5,

                             ### Fee ###
                             # Target mana utilization per block as a fraction of maximum
                             RELATIVE_TARGET_MANA_PER_BLOCK=0.5,
                             BLOBS_PER_BLOCK=3,  # fixed
                             L1_GAS_TO_VERIFY=1_000_000,  # fixed
                             L1_GAS_TO_PUBLISH=150_000,  # fixed
                             L1_BLOBGAS_PER_BLOB=int(2 ** 17),  # fixed, ~131k
                             POINT_EVALUATION_PRECOMIPLE_GAS=50_000,  # fixed
                             MINIMUM_MULTIPLIER_CONGESTION=1_000_000,  # sweep
                             # Maximum ratio for congestion multiplier
                             # roughly equivalent to a max increase of 22026x
                             MAXIMUM_MULTIPLIER_CONGESTION_RATIO=10,
                             RELATIVE_UPDATE_FRACTION_CONGESTION=0.5,  # sweep
                             # Maximum relative change L2 block to L2 block in congestion multiplier
                             MAX_RELATIVE_CHANGE_CONGESTION=0.03,
                             # Maximum percentage update for control parameters
                             MAXIMUM_UPDATE_PERCENTAGE_C=0.03,
                             # Initial proving cost in USD per transaction (control parameter)
                             PROVING_COST_INITIAL_IN_USD_PER_TX_C=0.03,
                             M_PROVER_TARGET = 3,

                             ### Reward ###
                             # See block reward design for more details
                             # Volatility factor for block rewards
                             BLOCK_REWARD_VOLATILITY=3.17e-6,
                             # Decay rate for block reward drift
                             BLOCK_REWARD_DRIFT_DECAY_RATE=1e-2,
                             # Speed adjustment factor for block rewards
                             BLOCK_REWARD_SPEED_ADJ=5.5e-4,
                             BLOCK_REWARD_SHARE_PROVER=0.5,

                             ### Staking ###
                             # Minimum stake required to be a validator
                             MIN_STAKE=DEFAULT_MIN_STAKE,  # fixed
                             # Size of the validator committee
                             VALIDATOR_COMMITTEE_SIZE=128,  # min 128, likely around 300
                             # Fraction of signatures needed for acceptance by L1 contract
                             SIGNATURES_NEEDED=0.5,
                             PERCENTAGE_STAKE_SLASHED_C=1.0,

                             ### Behavioural ###
                             # Average number of transactions per slot, used to calculate actual transactions per slot
                             AVERAGE_TX_COUNT_PER_SLOT=360,
                             # Sequencer modification to proving cost
                             PROVING_COST_MODIFICATION_E=0.0,
                             # Sequencer modification to fee-juice price
                             FEE_JUICE_PRICE_MODIFICATION_E=0.0,
                             # Frequency of oracle updates by Sequencers (exogenous)
                             ORACLE_UPDATE_FREQUENCY_E=0.5,
                             # Mean of the fee-juice price distribution, used to vary exchange rate
                             JUICE_PER_GWEI_MEAN=1.1e-6,
                             # Covariance of the fee-juice price distribution, used to vary exchange rate
                             JUICE_PER_GWEI_COV=0.10,
                             # Initial inflation estimate, used by users to determine max fee
                             INITIAL_INFLATION_ESTIMATE=0.0,
                             # Maximum relative mean of fee-juice price distribution, used by users to determine max fee
                             MAX_FEE_INFLATION_RELATIVE_MEAN=1.0,
                             # Maximum relative standard deviation of fee-juice price distribution, used by users to determine max fee
                             MAX_FEE_INFLATION_RELATIVE_STD=0.1,

                             ### L1 Pricing ###
                             # Mean price of gwei per L1 gas, used to construct synthetic L1 gas price
                             GWEI_PER_L1GAS_MEAN=30,
                             # Covariance of the gwei per L1 gas distribution, used to construct synthetic L1 gas price
                             GWEI_PER_L1GAS_COV=0.01,
                             # Mean price of gwei per L1 blob gas, used to construct synthetic L1 blob gas price
                             GWEI_PER_L1BLOBGAS_MEAN=5,
                             # Covariance of the gwei per L1 blob gas distribution, used to construct synthetic L1 blob gas price
                             GWEI_PER_L1BLOBGAS_COV=0.05,
                             # Threshold for sequencer to decide L1 gas price is not profitable anymore so won't continue their actions (exogenous)
                             SEQUENCER_L1_GAS_PRICE_THRESHOLD_E=float('nan'),
                             # Total mana multiplier is used to estimate actual mana per transaction, multiplied by overhead mana per transaction
                             TOTAL_MANA_MULTIPLIER_E=1.0,
                             # Probability of validator not providing signature, used to estimate actual number of signatures per block
                             SIGNATURE_SKIP_PROBABILITY=0.0,
                             # Maximum percentage of validators that will be slashed, used to construct synthetic mass slashing events
                             MAX_VALIDATORS_TO_SLASH=0.0,

                             # Scenario for Juice to Gwei exchange rate, used to construct synthetic timeseries of exchange rate for shock scenarios. Can be set to strictly increasing, stochastic, constant, etc.
                             JUICE_PER_GWEI_SCENARIO=JuiceGweiExchangeRateScenario.Stochastic,

                             # Arrival Processes for events related to the Block & Epoch proposal / acceptance
                             SHAPE_E_BLOCK_PROPOSE=2.0,
                             SCALE_E_BLOCK_PROPOSE=1/5,
                             SHAPE_E_BLOCK_VALIDATE=2.0,
                             SCALE_E_BLOCK_VALIDATE=1/3,
                             SHAPE_E_BLOCK_SENT=2.0,
                             SCALE_E_BLOCK_SENT=1/10,
                             PROBABILITY_E_EPOCH_FINISH=0.15,
                             EXPECTED_PROOFS_LAST_EPOCH_PER_TS=1,
                             EXPECTED_PROOFS_CURRENT_EPOCH_PER_TS=1,
                             MIN_FRACTION_COMPLETE_PROOF_LAST_EPOCH=0.6,
                             MAX_FRACTION_COMPLETE_PROOF_LAST_EPOCH=1.0,

                             # Exogenous market price of ETH in USD
                             market_price_eth=3300,
                             )
