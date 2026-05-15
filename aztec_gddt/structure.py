"""
Model structure definition for the Aztec GDDT simulation.

This module defines the sequential (partial) state update blocks of the cadCAD simulation model for the Aztec GDDT v2.
It organizes the simulation into logical blocks, each representing a different aspect of the system's behavior:

1. Time Tracking - Handles the advancement of time (L1 blocks) in the simulation
2. Set Exogenous Variables - Updates market-driven external values
3. Oracles & Scoring Functions - Simulates how oracles update their values based on market prices
4. Contract Values - Updates on-chain contract state values like congestion multiplier
5. Instantaneous metrics - Calculates derived metrics like base fee
6. Epoch/Slot evolution - Manages the core epoch and slot progression
7. Epoch Proving - Handles the proving and finalization of epochs

Each block defines:
- policies: Functions that generate signals based on current state (p_* functions)
- variables: Functions that update state variables based on signals (s_* functions)

The structure defined here is used by experiment.py to run simulations.
"""

from aztec_gddt.logic import *
from aztec_gddt.mechanism_functions import compute_base_fee, compute_base_fee_congestionless, compute_base_fee_proverless
from copy import deepcopy

# Define the raw model blocks with their policies and variables
# These form the structure of the cadCAD simulation model
RAW_MODEL_BLOCKS: list[dict] = [
    {
        'label': 'Time Tracking',
        'ignore': False,
        'desc': 'Updates the time in the system',
        'policies': {
            'evolve_time': p_evolve_time  # Policy that determines how time advances
        },
        'variables': {
            'l1_blocks_passed': s_blocks_passed,  # Update total L1 blocks
            'delta_l1_blocks': s_delta_blocks    # Update blocks passed in this step
        }
    },
    {
        'label': 'Set Exogenous Variables',
        'policies': {
        },
        'variables': {
            # Market prices evolve according to their respective models
            'market_price_juice_per_gwei': s_market_price_juice_per_gwei,
            'market_price_l1_gas': s_market_price_l1_gas,
            'market_price_l1_blobgas': s_market_price_l1_blobgas,
        }
    },
    {
        'label': 'Oracles & Scoring Functions',
        'ignore': False,
        'policies': {
            # Oracle update policies - determine when and how oracle values change
            'juice_per_gwei': p_oracle_juice_per_gwei,
            'l1_gas': p_oracle_l1_gas,
            'l1_blobgas': p_oracle_l1_blobgas,
            'proving_cost': p_oracle_proving_cost

        },
        'variables': {
            # State variables for oracle prices and their last update times
            'oracle_price_juice_per_gwei': replace_suf,
            'oracle_price_l1_gas': replace_suf,
            'oracle_price_l1_blobgas': replace_suf,
            'update_time_oracle_price_juice_per_gwei': replace_suf,
            'update_time_oracle_price_l1_gas': replace_suf,
            'update_time_oracle_price_l1_blobgas': replace_suf,
            'oracle_proving_cost': replace_suf

        }
    },
    {
        'label': 'Contract Values',
        'ignore': False,
        'policies': {
            # No policy functions needed - contract values are derived from state
        },
        'variables': {
            # Update contract state based on system conditions
            # Update fee multiplier based on network congestion
            'congestion_multiplier': s_congestion_multiplier
        }
    },
    {
        'label': 'Instantaneous metrics',
        'ignore': False,
        'policies': {
            # No policy functions needed - metrics are derived from state
        },
        'variables': {
            # Calculate base fee based on current state (using lambda for direct calculation)
            'base_fee': lambda p, _2, _3, s, _5: ('base_fee', compute_base_fee(p, s)),
            'base_fee_congestionless': lambda p, _2, _3, s, _5: ('base_fee_congestionless', compute_base_fee_congestionless(p, s)),
            'base_fee_proverless': lambda p, _2, _3, s, _5: ('base_fee_proverless', compute_base_fee_proverless(p, s)),
        }
    },
    {
        'label': 'Epoch/Slot evolution',
        'ignore': False,
        'policies': {
            # Core policy for epoch and slot lifecycle management
            # Handles slot events, transaction inclusion, and new slot/epoch creation
            'evolve_epoch_slot': p_epoch
        },
        'variables': {
            # Update epoch-related state variables
            'current_epoch': replace_suf,  # Replace with updated epoch
            'last_epoch': replace_suf,     # Update last epoch
            # Cumulative metrics are added to existing values
            'cumm_dropped_tx': add_suf,    # Count dropped transactions
            'cumm_excl_tx': add_suf,       # Count excluded transactions
            'cumm_total_tx': add_suf,      # Count total transactions
            'excess_mana': replace_suf,    # Update excess mana
            'l2_blocks_passed': add_suf,   # Count L2 blocks processed
            'cumm_blocks_with_collected_signatures': add_suf,  # Count blocks with signatures
            # Count blocks with sufficient signatures
            'cumm_blocks_with_enough_signatures': add_suf
        }
    },
    {
        'label': 'Epoch Proving',
        'ignore': False,
        'policies': {
            # Policy for epoch proving and finalization
            # Handles prover selection, quotes, and epoch finalization
            'epoch_proving': p_pending_epoch_proof
        },
        'variables': {
            # Update proving-related state variables
            'current_epoch': replace_suf,
            'last_epoch': replace_suf,             # Update last epoch state
            'last_reward_time_in_l1': replace_suf,  # Update when reward was last given
            'last_reward': replace_suf,            # Update reward amount
            'last_reward_to_provers': replace_suf,
            'last_reward_per_prover': replace_suf,
            # Cumulative metrics
            'cumm_empty_blocks': add_suf,          # Count empty blocks
            'cumm_unproven_epochs': add_suf,       # Count epochs without proofs
            # Count resolved epochs (finalized or reorged)
            'cumm_resolved_epochs': add_suf,
            'cumm_finalized_epochs': add_suf,      # Count successfully finalized epochs
            'cumm_mana_used_on_finalized_blocks': add_suf,  # Total mana in finalized blocks
            'cumm_fee_paid': add_suf,
            'cumm_fee_burnt': add_suf,
            'cumm_fee_to_provers': add_suf,
            'cumm_fee_to_sequencers': add_suf,
            'cumm_finalized_blocks': add_suf,      # Count finalized blocks
            'agents': replace_suf,
            'slash_count': replace_suf,
            'slash_amount': replace_suf
        }
    }
]

# Process the raw model blocks to create executable cadCAD blocks
# This creates concrete update functions for each variable
blocks: list[dict] = []
for block in [b for b in RAW_MODEL_BLOCKS if b.get("ignore", False) != True]:
    # Create a deep copy to avoid modifying the original
    _block: dict = deepcopy(block)

    # For each variable in the block, create the appropriate state update function
    for variable, suf in block.get("variables", {}).items():  # type: ignore
        if suf == add_suf:
            # For cumulative metrics, create an add function bound to this variable
            _block["variables"][variable] = add_suf(variable)  # type: ignore
        elif suf == replace_suf:
            # For state variables, create a replace function bound to this variable
            _block["variables"][variable] = replace_suf(
                variable)  # type: ignore

    blocks.append(_block)

# Create the final model blocks by filtering out any ignored blocks
# These blocks will be used directly by cadCAD for simulation execution
MODEL_BLOCKS = [block for block in blocks
                if block.get('ignore', False) is False]
