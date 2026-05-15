"""
Type definitions for the Aztec GDDT v2 model.

This module contains all the type annotations, dataclasses, and type definitions used
throughout the simulation.
"""

from typing import Annotated, TypedDict, Union, Optional
from dataclasses import dataclass, field
from enum import Enum, auto


class JuiceGweiExchangeRateScenario(Enum):
    """Defines possible scenarios for how the Juice/Gwei exchange rate evolves over time."""
    StrictlyIncreasing = auto()
    StrictlyDecreasing = auto()
    Constant = auto()
    Stochastic = auto()


# Time and block related types
Days = Annotated[float, 'days']  # Number of days
BlocksL1 = int
BlocksL2 = int
AgentUUID = str

Percentage = float

Token = float
ETH = float
Fiat = float

Gwei = Annotated[float, 'gwei']  # 1e-9 ETH
Gas = Annotated[int, 'gas']
GweiPerGas = Annotated[float, 'gwei/gas']
GweiPerMana = Annotated[float, 'gwei/mana']
JuicePerGwei = Annotated[float, 'juice/gwei']
JuicePerMana = Annotated[float, 'juice/mana']

USDPerETH = Annotated[float, 'usd/eth']

# Aztec-specific types
Juice = Annotated[float, 'juice']  # Aztec's analogue to Gwei
Mana = Annotated[int, 'mana']  # Aztec's analogue to Gas


@dataclass
class Transaction():
    """Represents an L2 transaction with its associated mana and fee values."""
    mana: Mana
    # max_fee_per_mana: int
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
    has_collected_signatures: bool = False
    has_block_header_on_l1: bool = False
    tx_count: int = 0
    tx_total_mana: Mana = 0
    tx_total_fee: Juice = 0

    @property
    def is_valid_proposal(self):
        """Determines if the proposal satisfies all validity conditions."""
        is_valid = self.has_proposal_on_network
        is_valid &= self.has_validator_signatures
        is_valid &= self.has_block_header_on_l1


@dataclass
class Epoch():
    init_time_in_l1: int
    validators: list[AgentUUID]
    slots: list[Slot]
    time_until_E_EPOCH_FINISH: BlocksL1
    pending_time_in_l1: int = -999  # Time in L1 when Epoch has entered Pending Chain
    finalized_time_in_l1: int = -999
    accepted_provers: set[AgentUUID] = field(default_factory=set)
    submitted_proofs: list[tuple[AgentUUID, BlocksL2]] = field(default_factory=list) # Each tuple represents a partial or complete proof.
    reward: Juice = float('nan')
    fee_compensation: Juice = float('nan')
    finalized: bool = False
    reorged_full: bool = False
    reorged_partial: bool = False


@dataclass
class Agent():
    uuid: AgentUUID
    stake: ETH
    score: float


class ModelState(TypedDict):
    timestep: int
    l1_blocks_passed: BlocksL1
    l2_blocks_passed: BlocksL2
    delta_l1_blocks: BlocksL1
    agents: dict[AgentUUID, Agent]
    validator_set: set[AgentUUID]
    current_epoch: Epoch
    last_epoch: Epoch

    # Block Reward related values
    last_reward: Juice
    last_reward_to_provers: Juice
    last_reward_per_prover: Juice
    last_reward_time_in_l1: BlocksL1

    # Market & Oracle Values
    market_price_juice_per_gwei: JuicePerGwei
    market_price_l1_gas: GweiPerGas
    market_price_l1_blobgas: GweiPerGas

    oracle_price_juice_per_gwei: JuicePerGwei
    oracle_price_l1_gas: GweiPerGas
    oracle_price_l1_blobgas: GweiPerGas
    oracle_proving_cost: GweiPerMana

    update_time_oracle_price_juice_per_gwei: BlocksL1
    update_time_oracle_price_l1_gas: BlocksL1
    update_time_oracle_price_l1_blobgas: BlocksL1

    congestion_multiplier: float
    excess_mana: Mana

    # State Metrics including cumulative values
    base_fee: JuicePerMana
    base_fee_congestionless: JuicePerMana
    base_fee_proverless: JuicePerMana
    cumm_empty_blocks: int
    cumm_unproven_epochs: int
    cumm_dropped_tx: int
    cumm_excl_tx: int
    cumm_total_tx: int
    cumm_resolved_epochs: int
    cumm_finalized_epochs: int
    cumm_mana_used_on_finalized_blocks: Mana
    cumm_finalized_blocks: BlocksL2
    cumm_blocks_with_collected_signatures: int
    cumm_blocks_with_enough_signatures: int
    cumm_fee_paid: Juice
    cumm_fee_burnt: Juice
    cumm_fee_to_provers: Juice
    cumm_fee_to_sequencers: Juice
    slash_count: int
    slash_amount: Juice


class ModelParams(TypedDict):
    label: str
    timestep_in_l1_blocks: int
    N_timesteps: int

    ### General ###
    OVERHEAD_MANA_PER_TX: Mana  # sweep 1k, 10k, 20k or 50k
    MAXIMUM_MANA_PER_BLOCK: Mana  # sweep 20m or 40m
    SIM_TST_TOTAL_SUPPLY: Token  # Assigned
    SIM_LAUNCH_VALUATION: Fiat  # Assigned
    L2_SLOTS_PER_L2_EPOCH: BlocksL2  # fixed
    L1_SLOTS_PER_L2_SLOT: BlocksL1  # fixed
    MIN_ORACLE_UPDATE_LAG_C: BlocksL1

    ### Fee ###
    RELATIVE_TARGET_MANA_PER_BLOCK: Percentage  # sweep, relative to the maximum mana per block
    BLOBS_PER_BLOCK: int
    L1_GAS_TO_VERIFY: Gas
    L1_GAS_TO_PUBLISH: Gas
    L1_BLOBGAS_PER_BLOB: Gas
    POINT_EVALUATION_PRECOMIPLE_GAS: Gas
    MINIMUM_MULTIPLIER_CONGESTION: float # sweep
    MAXIMUM_MULTIPLIER_CONGESTION_RATIO: float 
    RELATIVE_UPDATE_FRACTION_CONGESTION: float # sweep, relative to the maximum mana per block
    MAX_RELATIVE_CHANGE_CONGESTION: Percentage
    MAXIMUM_UPDATE_PERCENTAGE_C: Percentage
    PROVING_COST_INITIAL_IN_USD_PER_TX_C: float
    M_PROVER_TARGET: float

    ### Reward ###
    BLOCK_REWARD_VOLATILITY: float  # sweep
    BLOCK_REWARD_DRIFT_DECAY_RATE: float  # sweep
    BLOCK_REWARD_SPEED_ADJ: float  # sweep
    BLOCK_REWARD_SHARE_PROVER: Percentage # set by governance

    MIN_STAKE: ETH  # fixed
    VALIDATOR_COMMITTEE_SIZE: int  # min 128, likely around 300
    SIGNATURES_NEEDED: Percentage
    PERCENTAGE_STAKE_SLASHED_C: Percentage

    ### Behavioural ###
    AVERAGE_TX_COUNT_PER_SLOT: int
    PROVING_COST_MODIFICATION_E: Percentage  # env, sweep
    FEE_JUICE_PRICE_MODIFICATION_E: Percentage  # env, sweep
    ORACLE_UPDATE_FREQUENCY_E: Percentage  # env, sweep
    JUICE_PER_GWEI_MEAN: JuicePerGwei
    JUICE_PER_GWEI_COV: JuicePerGwei
    INITIAL_INFLATION_ESTIMATE: Percentage
    MAX_FEE_INFLATION_RELATIVE_MEAN: Percentage
    MAX_FEE_INFLATION_RELATIVE_STD: Percentage

    GWEI_PER_L1GAS_MEAN: GweiPerGas
    GWEI_PER_L1GAS_COV: Percentage
    GWEI_PER_L1BLOBGAS_MEAN: GweiPerGas
    GWEI_PER_L1BLOBGAS_COV: Percentage
    SEQUENCER_L1_GAS_PRICE_THRESHOLD_E: Gwei
    TOTAL_MANA_MULTIPLIER_E: Percentage
    SIGNATURE_SKIP_PROBABILITY: Percentage
    MAX_VALIDATORS_TO_SLASH: Percentage
    JUICE_PER_GWEI_SCENARIO: JuiceGweiExchangeRateScenario

    # Arrival Processes for events related to the Block & Epoch proposal / acceptance
    SHAPE_E_BLOCK_PROPOSE: float
    SCALE_E_BLOCK_PROPOSE: float
    SHAPE_E_BLOCK_VALIDATE: float
    SCALE_E_BLOCK_VALIDATE: float
    SHAPE_E_BLOCK_SENT: float
    SCALE_E_BLOCK_SENT: float
    PROBABILITY_E_EPOCH_FINISH: float
    EXPECTED_PROOFS_LAST_EPOCH_PER_TS: float
    EXPECTED_PROOFS_CURRENT_EPOCH_PER_TS: float
    MIN_FRACTION_COMPLETE_PROOF_LAST_EPOCH: float
    MAX_FRACTION_COMPLETE_PROOF_LAST_EPOCH: float

    # Exogenous
    market_price_eth: USDPerETH
