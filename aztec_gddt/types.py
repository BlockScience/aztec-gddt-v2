from typing import Annotated, TypedDict, Union, Optional
from dataclasses import dataclass, field


Days = Annotated[float, 'days']  # Number of days
BlocksL1 = int
BlocksL2 = int
AgentUUID = str

Percentage = float

Token = float
ETH = float
Fiat = float

Wei = Annotated[int, 'wei'] # 1e-18 ETH
Gas = Annotated[int, 'gas'] 

Juice = Annotated[int, 'juice'] # Aztec's analogue to Wei
Mana = Annotated[int, 'mana'] # Aztec's analogue to Gas

@dataclass
class Transaction():
    mana: Mana
    max_fee_per_mana: int
    priority_fee_per_mana: int



@dataclass
class Slot():
    init_time_in_l1: int
    proposer: AgentUUID
    time_until_E_BLOCK_PROPOSE: float
    time_until_E_BLOCK_VALIDATE: float
    time_until_E_BLOCK_SENT: float
    has_proposal_on_network: bool = False
    has_validator_signatures: bool = False
    has_block_header_on_l1: bool = False


    @property
    def is_valid_proposal(self):
        is_valid = self.has_proposal_on_network
        is_valid &= self.has_validator_signatures
        is_valid &= self.has_block_header_on_l1

@dataclass
class Epoch():
    init_time_in_l1: int
    validators: list[AgentUUID]
    slots: list[Slot]
    time_until_E_EPOCH_QUOTE_ACCEPT: BlocksL1
    time_until_E_EPOCH_FINISH: BlocksL1
    pending_time_in_l1: int = -999# Time in L1 when Epoch has entered Pending Chain
    finalized_time_in_l1: int = -999
    prover_quotes: dict[AgentUUID, Token] = field(default_factory=dict)
    accepted_prover: Optional[AgentUUID] = None
    accepted_prover_quote: Token = float('nan')
    reward: Token = float('nan')
    fee_compensation: Token = float('nan')
    finalized: bool = False
    reorged: bool = False


@dataclass
class Agent():
    uuid: AgentUUID
    commitment_bond: Token
    score: float


class ModelState(TypedDict):
    l1_blocks_passed: BlocksL1
    delta_l1_blocks: BlocksL1
    agents: list[Agent]
    validator_set: set[AgentUUID]

    PROVING_COST_MODIFIER: float

    current_epoch: Epoch
    last_epoch: Epoch

    last_reward: Token
    last_reward_time_in_l1: BlocksL1



@dataclass
class GeneralParams():
    OVERHEAD_MANA_PER_TX: Mana # sweep 1k, 10k, 20k or 50k
    MAXIMUM_MANA_PER_BLOCK: Mana # sweep 20m or 40m
    TST_TOTAL_SUPPLY: Token # Assigned
    LAUNCH_VALUATION: Fiat # Assigned
    L2_SLOTS_PER_L2_EPOCH: BlocksL2 = 32 # fixed
    L1_SLOTS_PER_L2_SLOT: BlocksL1 = 3  # fixed
    PROVER_SEARCH_PERIOD: BlocksL2 = 13 # fixed


@dataclass
class FeeParams():
    TARGET_MANA_PER_BLOCK: Mana # sweep, ~0.5 of GeneralParams.MAXIMUM_MANA_PER_BLOCK
    BLOBS_PER_BLOCK: int = 3 # fixed
    L1_GAS_TO_VERIFY: Gas = 1_000_000 # fixed
    L2_SLOTS_PER_EPOCH: int = 32 # fixed
    L1_GAS_TO_PUBLISH: Gas = 150_000 # fixed


@dataclass
class RewardParams():
    BLOCK_REWARD_VOLATILITY: float # sweep
    BLOCK_REWARD_DRIFT_DECAY_RATE: float # sweep
    BLOCK_REWARD_SPEED_ADJ: float # sweep



@dataclass
class StakingParams():
    MIN_STAKE: ETH = 32 # fixed
    VALIDATOR_COMMITTEE_SIZE: int = 128 # min 128, likely around 300
    SIGNATURES_NEEDED: Percentage = 0.5


@dataclass
class SlashingParams():
    BOND_SIZE: Token # control, sweep
    BOND_SLASH_PERCENT: Percentage = 1.0 # control, fixed
    

@dataclass
class ProvingParams():
    PROVING_COST_MODIFICATION_E: Percentage # env, sweep
    MAXIMUM_PROVING_COST_WEI_PER_MANA_PERCENT_CHANGE_PER_L2_SLOT: Percentage # control, sweep
    MAX_BASIS_POINT_FEE: Percentage = 0.9 # control, fixed


class ModelParams(TypedDict):
    label: str
    timestep_in_l1_blocks: int

    general: GeneralParams
    fee: FeeParams
    reward: RewardParams
    stake: StakingParams
    slash: SlashingParams




