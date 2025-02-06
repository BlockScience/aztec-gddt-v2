# aztec_gddt_v2

## Introduction

This repository contains a cadCAD simulation model of the Aztec system under the new specification, where 32 slots are bundled per epoch and proven collectively. If proven, they are then added to the "proven chain" as a batch. 


The goal of the model is to understand the effect of various design parameters,  mechanisms and agent behaviors on the health of the Aztec network, as measured through various Key Performance Indicators (KPIs). These KPIs focused on three main scenario groups aligned with the Aztec Labs team, namely the fee mechanism, staking & slashing and L2 congestion.

### What Is In This Document?

This document provides:
* an overview of the system under consideration
* an overview of the experiment setup
* mathematical and software specifications for the cadCAD simulation model for the system, and
* information on how to use the model

### What This Model Enables

Using the model, it is possible to run simulations and track the effect of various parameter values on final metrics and KPIs of interest. Possibilities include parameter sweeps and their impact on KPIs, A/B testing, and visualization of trajectories. 

## Model Overview

The simulation model is written in cadCAD. In this section we will record the overall structure of the cadCAD model. We note that an overview of the implemented Aztec System is found in [current specs, Nov. 2024](https://hackmd.io/mgRskzOcTHycxwtcxn4nsA?view#Current-Specs-Description---Nov-2024).

There are five important aspects of a cadCAD model:
* Model Variables
* Model Parameters
* Partial State Update Blocks
* Policies
* State Update Functions

For more general information about the purpose each piece serves in a cadCAD model, please see [this overview](https://github.com/cadCAD-org/cadCAD/blob/master/documentation/README.md). 

### Model Variables

All variables of the Aztec system recorded in the model are implemented in a `ModelState` class, which can be accessed in `types.py`. 
In general, the model variables record important system states, including: the current stage of the process,agents engaged with the system, number of finalized epochs and blocks produced, base fee, among others. 

### Model Parameters

Parameters represent aspects of the simulation which are fixed before the beginning of a single model run. In this model, parameters are implemented in the `ModelParams` class, available in `types.py`. There are *endogeneous or control parameters*, corresponding to aspects of the system under Aztec's control, such as RELATIVE_TARGET_MANA_PER_BLOCK, or MAXIMUM_MANA_PER_BLOCK. In addition, the model includes *exogeneous parameters* which correspond to necessary assumptions about agent behavior (*behavioral parameters*), such as the probability that a validator accepts a prover quote or assumptions about the environment (*environmental parameters*) such as network congestion and L1 gas prices. 

### Model Mechanisms

#### Block Reward
A sequencer receives a reward R from the protocol for every block they proposed, if the epoch that includes it is finalized. The sequencer also receives compensation in the way of transactions fees F for all transactions in their block, when an epoch is finalized (that users submitting transactions pay when their transactions are included in a block). All of the rewards are shared with the Prover of the Epoch, where the share is defined by the accepted Prover quote. 

Over time, i.e. as the Aztec network becomes established, the block reward should become less impactful--it is used primarily as insurance for sequencers while the network is less mature, and should become a less important part of sequencer revenue after the "bootstrapping" phase of network growth. Thus, over time the block reward should decline.

Prior experience has shown, however, that a deterministic block reward, even if declining over time, has an undesirable incentive side-effect: participants will anchor reward expectations on said block reward, instead of seeking to make revenue primarily through transactions fees - i.e. real useage of the network. These participants might then orient and centralize their operations to such an extent that they can capture a consistent share of the (ever-shrinking) block reward.

This negative externality can be mitigated by placing the block reward on the same footing as transactions fees, i.e. by providing a source of randomness to the block reward. This extends a sequencer's risk management problem, incentivizing activities that ensure a consistent share of transactions fees. Provided that transactions fees are themselves growing (in a way that can be modeled), a random and declining block reward can be eventually discarded by a sequencer as the network matures.

#### Transaction Fee Mechanism - Aztec's proposal
A transaction incurs the following costs:
1. Proposers/validators validate and simulate the transaction 
1. Proposers publish pending blocks to L1 (represented through "L1 Gas to publish proposal")
    2. Includes publishing content as blobs to DA (represented through "L1 DA Cost" and "L1 Gas for DA")
3. Proposers/validators/nodes update their state
4. Provers generate the avm proofs for the transaction (represented through "Proving Cost per mana")
5. Provers generate the rollup proofs for the blocks/epoch (represented through "Proving Cost per mana")
6. Provers verify the epoch proof on L1 (represented through "L1 Gas to verify Block")

A transaction is also "metered", which can be done both by users and by proposers (and anyone else with the required data). This returns the amount of "mana" it spends, similar to gas on ETH.
Users pay for their transactions as:
$transaction\_fee = mana\_consumed * base\_fee\_per\_mana + (optional) mana\_consumed * priority\_fee\_per\_mana$

The transaction fee mechanism incorporates various components to reflect the costs listed above. These components include for example the fees for publishing transactions on L1 (including using blobs for DA), the costs to prove a transaction (and its epoch), and congestion pricing. 


#### Lag and Secondary Market Module
The transaction fee mechanism mentioned above relies on various exogenous factors - such as ETH gas prices, blob gas prices, proving costs and exchange rates. These are available to users through oracles and are as such not tightly fixed to their real values. As such, these oracle values might lag behind the current real value, until the value is updated. Additionally, ETH gas and blobgas price oracles are currently specified with a lifetime value of 5 - i.e. any oracle value is in effect for at least 5 L2 slots. 
Considering that (blob)gas prices and exchange rates (necessary for the computation of the fee mechanism) can show high volatility at times, it is considered a priority to understand how such lags - especially under high volatility - effect the base fee. As such, the model includes both lags and secondary market simplifications to reflect such environmental factors. 

## Experiment Overview
A *scenario group* refers to a set of experiments where the goal is to understand the impact of pre-defined random variables or uncertain parameters on metrics of interest.  Each scneario group is classified within one of 4 major categories: fee mechanism, staking/slashing, L2 congestion, and block reward. In total, we have identified 9 scenario groups, as follows:

- ====Fee Mechanism====
    - Volatility
    - L2 Cost Censorship
    - Shock Analysis
    - Oracle Sensitivity
- ====Staking/Slashing====
    - Resumption from Inactivity 
    - Validator Ejection
- ====L2 Congestion====

Because an exhaustive 'sweep' of every possible combination of relevant protocol parameters is computationally infeasible, the *methodology* used in this study instead performes **an adaptive search**, whereby a coarse initial grid of parameters is successively refined by applying the success criteria to generated KPIs, and inferring a new 'direction' of search for a succeeding grid. Convergence is achieved when all success criteria are met across the performed simulations. Although it is always possible that multiple "equilibria" exist, such that success criteria are met by parameter combinations that are not found from adaptive search, the initial grid is informed by both intuitive ranges and existing parameter values from the Aztec network and hence benefits from the expert knowledge used to define those initial values.

Future work can perform a more thorough search of the underlying parameter space, in addition to performing more demand scenarios and realizations from the exogenous distributions that represent external factors.

Description and discussion of the computational experiments **for each** scenario group is structured as follows: (1) objective, (2) experimental setup, (3) computational complexity, (4) adaptive grid results, (5) protocol parameter recommendations, (6) decision tree and parameter importance, (7) parameter impact on metrics and (8) conlusion. This structure is followed by the notebooks where the experimental results are discussed, see for instance [scenarios-L2-congestion](https://github.com/BlockScience/aztec-gddt-v2/blob/main/notebooks/scenarios_L2_congestion.ipynb).

In what follows, we define the high-level content of each of the previous 7 sections. Details can be found in each dedicated scenario group notebook. 

:::spoiler **Objective**

The *objective* of a computational experiment, regarding a specific scenerio group, is to gather valuable insights around targeted goals and guide important business decisions, such as the sensitivity of the transactions fee to the rollup contract parameters determined by one or more outside 'oracles', or the cost for an attacker to control up to x% of the validator committee.  

:::

        
:::spoiler **Experimental Setup**

- **Simulation input/output per Monte Carlo run**: Input corresponds to a parameter, list of parameters or simulated time series of the uncertain parameter(s) affecting the experiment under study. Output corresponds to a system variable(s) useful to determine the value of the experiment's specified metric(s).

- **Sweep parameters**: Set of uncertain parameters taking on a user--specified value at each simulation run, referred to as sweep parameters. These parameters jointly determine/affect the simulation input needed to test a scenario group.  Such parameters can be divided into two categories: control and environmental.

    - Control: An enumerated list of the protocol parameters is given, using their `name` in the code and any relevant abbreviation. Additionally, a table with the following information is also provided.

    
        | Full Name | Sweep Variable Name | Sweep Values | Units |
        | -------- | -------- | -------- | -------- |
        | Text     | Text     | Value     | Text
        
    - Environmental: An enumerated list of the environmental parameters is given, using their `name` in the code and any relevant abbreviation. Additionally, a table with the following information is also provided.

    
        | Full Name | Sweep Variable Name | Sweep Values | Units |
        | -------- | -------- | -------- | -------- |
        | Text     | Text     | Value     | Text
   

- **Simulation behavior**: Rule or set of rules implemented in the simulation reflecting the actions taken by agents (e.g., a sequencer) or the state change of an object (e.g., transaction), conditional upon the ocurrence of a pre-defined event. For instance, a simulation behavior may be such that a sequencer will not post a block to L1 *if* the revenue from the block is less than the cost of posting the block.

- **Threshold Inequalities**: If any, a list of threshold inequalities is provided here--each list item contains the name of the inequality and the actual threshold values used in the scenarios. Each item's reference in the code, usually with a suffix `_success`, is given.

- **Metrics & Interpretation**: Metrics are designed depending on the objective of a scenario group. 
A list of Metrics or KPIs is provided here--each list item contains the KPI number, definition and brief description.
     
:::

:::spoiler **Computational Complexity**
Includes the following statistics:

    1. Total number of parameter constellations
    2. Total number of Monte Carlo runs per constellation
    3. Total number of experiments per adaptive grid
    4. Number of adaptive grid searches
    5. Total number of parameter constellations evaluated

:::

:::spoiler **Adaptive Grid Results**

The evolution of the parameter selection process is presented as a visualization, showing the convergence of the protocol parameter ranges as different success criteria are achieved. Both, control and environmental parameters were initialized for the first adaptive grid search according to discussions with the Aztec team, and BlockScience best practice.

:::

:::spoiler **Protocol Parameter Recommendations**
Based upon the adaptive grid results, the recommended parameter ranges are presented.
:::
::: spoiler **Decision Tree and Parameter Importance**
Using the adaptive grid results, a **decision tree** is built to infer the importance of different parameters on the associated KPI-based threshold inequalities. This provides a method of assessing whether one or more parameters are 'crucial' to success, in the sense that they have an outsized impact on the success criteria. This approach leverages decision trees that are fit to the results of the entire adaptive grid process.

To provide guidance on how to interpret such decision tree, below we include a formal definition.

**Decision Tree Classification**

A decision tree is a machine-learning-based classifier. Given the simulation results, for each threshold inequality the tree recursively associates different samples from the results, according to sorting criteria based upon one or more of the protocol parameters of the simulation.

Each decision tree below corresponds to one of the threshold inequalities stated above. Where the decision tree is 'empty', the threshold inequality was either 1) always fulfilled during the simulations, or 2) never fulfilled during the simulations. In this case no sensitivity analysis can be performed, as the threshold inequalities do not vary according to the different parameter combinations that were swept.

The title of the decision tree includes the threshold inequality under scrutiny, in addition to a technical 'score' (usually "100%") and the number of simulation results used as the dataset. Within the decision tree presented, each non-terminal 'node' is labeled with the following information:


1. The sorting variable used and its cutoff value used for classification, in the form of `parameter_name <= x` where `x` is the cutoff value. Branches to the left of this node indicate satisfaction of this inequality, while branches to the right indicate violations, i.e. `parameter_name > x`.
2. A Gini coefficient representing the method of recursive association used.
3. The total number of simulation results ("samples = y%") as a percentage "y" that are considered at this node.
4. The breakdown of the simulation results considered into the left and right branches ("value = [p, 1-p]"), where "p" is the fraction of results that satisfy the `parameter_name = x` constraint, and "1-p" the fraction satisfying `parameter_name > x`.
5. The classification of the majority of the simulation results at this node (note that this is not a final classification, as it appears in a non-terminal node, and can be arbitrary if the results are split equally across classes).

**Terminal** nodes ("leaves") represent the final classification of that proportion of the simulation results that arrive at the node, and have most of the same information as a non-terminal node, with the exception that there is no branching performed and hence no sorting variable displayed. Here the most important information is the classification (last line).

Non-terminal and terminal nodes colored in blue correspond to the threshold inequality being met, and by following blue boxes from a terminal node up to the root of tree a set of `parameter_name <= x` and/or `parameter_name > x` sorting criteria can be chained together.

Upon successful classification, it is usual for the terminal node to have a breakdown "value = [1.0, 0.0]" or "value = [0.0, 1.0]", indicating that 100% of the remaining simulation results treated are either satisfying the threshold inequality under treatment (left value is 1.0), or not satisfying the threshold inequality (right value is 1.0).

For further information regarding the decision tree approach adopted here please see the [Decision Trees](https://scikit-learn.org/stable/modules/tree.html#) documentation from the `scikit-learn` library.


:::
::: spoiler **Parameter Impact on Metrics**
A density approach (histogram) can be used to assess the impact of protocol parameters on the KPIs of the scenario. The KPI densities are shown for each protocol parameter sweep value, providing a visual indication of the impact of the parameter on the density shape and location. 
:::

::: spoiler **Conclusion**
An overall assessment of the scenario results is provided, highlighting any problems, caveats, implications and possibilities for future/extended work.
:::

## Technical Details 

In this section we describe how to run the model on their own computer, if desired. 

### How to Install cadCAD

#### 1. Pre-installation Virtual Environments with [`venv`](https://docs.python.org/3/library/venv.html) (Optional):
It's a good package managing practice to create an easy to use virtual environment to install cadCAD. You can use the built in `venv` package. Note that this repo requires cadCAD 0.5, which is the latest version released December 2023. 

***Create** a virtual environment:*
```bash
$ python3 -m venv ~/cadcad
```

***Activate** an existing virtual environment:*
```bash
$ source ~/cadcad/bin/activate
(cadcad) $
```

***Deactivate** virtual environment:*
```bash
(cadcad) $ deactivate
$
```

#### 2. Installation: 
Requires [>= Python 3.11](https://www.python.org/downloads/) 

**Install Using [pip](https://pypi.org/project/cadCAD/)** 
```bash
$ pip3 install cadcad
```

**Install all packages with requirement.txt**
```bash
$ pip3 install -r requirements.txt
```

### Notebooks & Experiment Results

Reports with the experiment results for the above-mentioned scenarios can be found within the notebooks folder (see project directory structure below). As an example, consider `aztec-gddt-v2/notebooks/scenarios_L2_congestion.ipynb`. 

### Project Directory Structure

```
├── aztec_gddt: the `cadCAD` model as encapsulated by a Python Module
│   ├── __init__.py
│   ├── __main__.py
│   ├── default_params.py: System parameters
│   ├── experiment.py: Code for running experiments
│   ├── helper_types.py: Workflow for all scoped experiments
│   ├── logic.py: All logic for substeps
│   ├── mechanism_functions.py: All implemented mechanisms
│   ├── scenario_experiments.py: Workfllow for all scoped experiments
│   ├── structure.py: The PSUB structure
│   └── types.py: Types used in model
├── notebooks: Notebooks for aiding in development
├── tests: Tests for ensuring correct functionality
├── LICENSE
├── README.md
├── requirements.txt: Production requirements
```

