from aztec_gddt.types import *
from random import random



N_AGENTS = 300
DEFAULT_INITIAL_AGENTS = []
for i in range(N_AGENTS):
    a = Agent(uuid=str(i), 
              commitment_bond=5 * random(), 
              score=5 * random())
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
    agents=DEFAULT_INITIAL_AGENTS,
    validator_set=set(),
    PROVING_COST_MODIFIER=0.0,
    current_epoch=DEFAULT_INITIAL_EPOCH,
    last_epoch=DEFAULT_LAST_EPOCH
)

general_params = GeneralParams(OVERHEAD_MANA_PER_TX=10_000,
                              MAXIMUM_MANA_PER_BLOCK=100, # TODO check
                              TST_TOTAL_SUPPLY=1_000_000_000, # TODO check
                              LAUNCH_VALUATION=1_000_000_000 # TODO check
)

fee_params = FeeParams(TARGET_MANA_PER_BLOCK=int(general_params.MAXIMUM_MANA_PER_BLOCK * 0.5)
                       )


reward_params = RewardParams(BLOCK_REWARD_VOLATILITY=float('nan'), # TODO
                             BLOCK_REWARD_DRIFT_DECAY_RATE=float('nan'), # TODO
                             BLOCK_REWARD_SPEED_ADJ=float('nan') # TODO
                             )


stake_params = StakingParams()

slash_params = SlashingParams(BOND_SIZE=1)


DEFAULT_PARAMS = ModelParams(label='default',
                             timestep_in_l1_blocks=1,
                             general=general_params,
                             fee=fee_params,
                             reward=reward_params,
                             stake=stake_params,
                             slash=slash_params)