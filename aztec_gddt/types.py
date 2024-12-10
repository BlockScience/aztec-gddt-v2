from typing import Annotated, TypedDict, Union, Optional
from dataclasses import dataclass


Days = Annotated[float, 'days']  # Number of days
BlocksL1 = int
BlocksL2 = int
AgentUUID = int
Token = float

@dataclass
class Slot():
    proposer: AgentUUID
    has_proposal_on_network: bool = False
    has_content_on_network: bool = False
    has_validator_signatures: bool = False
    has_block_header_on_l1: bool = False

    @property
    def is_valid_proposal(self):
        is_valid = self.has_proposal_on_network
        is_valid &= self.has_content_on_network
        is_valid &= self.has_validator_signatures
        is_valid &= self.has_block_header_on_l1

@dataclass
class Epoch():
    init_time_in_l1: int
    validators: list[AgentUUID]
    slots: list[Slot]
    prover_quotes: dict[AgentUUID, Token] = dict()
    accepted_prover: Optional[AgentUUID] = None
    accepted_prover_quote: Token = float('nan')
    reward: Token = float('nan')
    fee_compensation: Token = float('nan')


@dataclass
class Agent():
    uuid: int
    commitment_bond: Token

class ModelState(TypedDict):
    l1_blocks_passed: BlocksL1
    delta_l1_blocks: BlocksL1
    epochs: list[Epoch]
    agents: list[Agent]

class ModelParams(TypedDict):
    label: str

