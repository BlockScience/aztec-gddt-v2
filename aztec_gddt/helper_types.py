from aztec_gddt.types import *
from aztec_gddt.default_params import *
from aztec_gddt.structure import MODEL_BLOCKS
from aztec_gddt.utils import policy_aggregator
from cadCAD.configuration import Experiment  # type: ignore
from cadCAD.configuration.utils import config_sim  # type: ignore
from cadCAD.tools.preparation import sweep_cartesian_product  # type: ignore
from random import sample
from dataclasses import dataclass, field
import numpy as np
from typing import Iterable
from dataclasses_json import dataclass_json

@dataclass
class ExperimentWrapper():
    params_swept_control: dict
    params_swept_env: dict
    N_timesteps: int
    N_samples: int
    N_configs: int
    experiment: Experiment
    label: str = ""

    def split_into_chunks(self, N_chunks) -> Iterable[Experiment]:
        splitted_configs: list[dict] = list(np.array_split(self.experiment.configs, N_chunks)) # type: ignore

        for i, splitted_config in enumerate(splitted_configs):
            exp = Experiment()
            exp.configs = list(splitted_configs)
            yield exp



@dataclass_json
@dataclass
class ExperimentParamSpec():
    params_swept_control: dict
    params_swept_env: dict
    N_timesteps: int
    N_samples: int
    N_config_sample: int
    relevant_per_trajectory_metrics: list[str] = field(default_factory=list)
    relevant_per_trajectory_group_metrics: list[str] = field(default_factory=list)
    label: str = ""

    @property
    def N_params(self) -> int:
        N_params = 1
        for k, v in self.params_swept_control.items():
            N_params *= len(v)
        for k, v in self.params_swept_env.items():
            N_params *= len(v)
        return N_params
    
    @property
    def N_trajectories(self) -> int:
        return self.N_params * self.N_samples
    
    @property
    def N_measurements(self) -> int:
        return self.N_trajectories * self.N_timesteps

    def print_control_params(self):
        for k, v in self.params_swept_control.items():
            print(f"{k}: {v}")

    def print_env_params(self):
        for k, v in self.params_swept_env.items():
            print(f"{k}: {v}")

    def prepare(self=1) -> ExperimentWrapper:
        default_params = {k: [v] for k, v in DEFAULT_PARAMS.items()}
        default_params['N_timesteps'] = [self.N_timesteps] # XXX

        params_to_sweep = {**default_params, **
                           self.params_swept_env, **self.params_swept_control}

        prepared_params = sweep_cartesian_product(params_to_sweep)

        states_list = [DEFAULT_INITIAL_STATE]

        exp = Experiment()
        for state in states_list:
            simulation_parameters = {"N": self.N_samples,
                                     "T": range(self.N_timesteps), "M": prepared_params}
            sim_config = config_sim(simulation_parameters)  # type: ignore
            exp.append_configs(
                sim_configs=sim_config,
                initial_state=state,
                partial_state_update_blocks=MODEL_BLOCKS,
                policy_ops=[policy_aggregator],
            )

        if int(self.N_config_sample) > 0:
            exp.configs = sample(exp.configs, int(self.N_config_sample))

        N_configs = len(exp.configs)

        wrapper = ExperimentWrapper(params_swept_control=self.params_swept_env,
                                    params_swept_env=self.params_swept_env,
                                    N_timesteps=self.N_timesteps,
                                    N_samples=self.N_samples,
                                    N_configs=N_configs,
                                    experiment=exp,
                                    label=self.label)
        return wrapper
