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
    last_reward_time_in_l1=0, # XXX
    last_reward=15_000, # XXX

    # Market & Oracle Values
    market_price_juice_per_mana = 5.0, # TODO
    market_price_l1_gas = 30, # TODO
    market_price_l1_blobgas = 5, # TODO
    oracle_price_juice_per_mana = 4.0, # TODO
    oracle_price_l1_gas = 28, # TODO
    oracle_price_l1_blobgas = 4, # TODO
    oracle_proving_cost = 0, # TODO
    congestion_multiplier = 1.0, # TODO
    excess_mana = 0, # TODO


    # State Metrics
    base_fee=float('nan'),
    cumm_empty_blocks=0,
    cumm_unproven_epochs=0,
    cumm_dropped_tx=0,
    cumm_excl_tx=0,
    cumm_resolved_epochs=0,
    cumm_finalized_epochs=0,
    cumm_mana_used_on_finalized_blocks=0,
    cumm_finalized_blocks=0
)

general_params = GeneralParams(OVERHEAD_MANA_PER_TX=10_000,
                              MAXIMUM_MANA_PER_BLOCK=20_000_000, # 20M or 40M
                              TST_TOTAL_SUPPLY=500_000_000, # XXX
                              LAUNCH_VALUATION=1_500_000_000 # XXX
)

fee_params = FeeParams(TARGET_MANA_PER_BLOCK=int(general_params.MAXIMUM_MANA_PER_BLOCK * 0.5),
                       UPDATE_FRACTION_CONGESTION=general_params.MAXIMUM_MANA_PER_BLOCK # HACK
                       )


reward_params = RewardParams(BLOCK_REWARD_VOLATILITY=3.17e-6,
                             BLOCK_REWARD_DRIFT_DECAY_RATE=1e-2, 
                             BLOCK_REWARD_SPEED_ADJ=5.5e-4 
                             )


stake_params = StakingParams()

slash_params = SlashingParams(BOND_SIZE=DEFAULT_BOND_SIZE)


behavioral_params = BehavioralParams(AVERAGE_TX_COUNT_PER_SLOT=360)

DEFAULT_PARAMS = ModelParams(label='default',
                             timestep_in_l1_blocks=1,
                             general=general_params,
                             fee=fee_params,
                             reward=reward_params,
                             stake=stake_params,
                             slash=slash_params,
                             behavior=behavioral_params)